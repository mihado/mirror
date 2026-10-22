use serde_json::{Map, Number, Value};

pub(crate) fn coerce_to_schema(value: &mut Value, schema: &Value) {
    *value = coerce_with_schema(value.take(), schema);
}

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind.as_str()],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn satisfies(value: &Value, schema: &Value) -> bool {
    let kinds = schema_types(schema);
    if !kinds.is_empty() && !kinds.iter().any(|kind| matches_type(value, kind)) {
        return false;
    }
    if let Some(Value::Array(members)) = schema.get("anyOf") {
        if !members.iter().any(|member| satisfies(value, member)) {
            return false;
        }
    }
    if let Some(Value::Array(members)) = schema.get("oneOf") {
        if members
            .iter()
            .filter(|member| satisfies(value, member))
            .count()
            != 1
        {
            return false;
        }
    }
    if let Some(Value::Array(members)) = schema.get("allOf") {
        if !members.iter().all(|member| satisfies(value, member)) {
            return false;
        }
    }
    if let Some(Value::Array(allowed)) = schema.get("enum") {
        if !allowed.contains(value) {
            return false;
        }
    }
    if let (Value::Object(object), Some(Value::Object(properties))) =
        (value, schema.get("properties"))
    {
        let required = required_names(schema);
        for (name, property) in properties {
            match object.get(name) {
                Some(field) if !satisfies(field, property) => return false,
                None if required.contains(&name.as_str()) => return false,
                _ => {}
            }
        }
    }
    if let (Value::Array(items), Some(item_schema @ Value::Object(_))) =
        (value, schema.get("items"))
    {
        if !items.iter().all(|item| satisfies(item, item_schema)) {
            return false;
        }
    }
    true
}

fn required_names(schema: &Value) -> Vec<&str> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

fn without_digit_separators(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut digits = String::with_capacity(text.len());
    for (index, byte) in bytes.iter().enumerate() {
        match byte {
            b'0'..=b'9' => digits.push(char::from(*byte)),
            b'_' if index > 0
                && bytes[index - 1].is_ascii_digit()
                && bytes.get(index + 1).is_some_and(u8::is_ascii_digit) => {}
            _ => return None,
        }
    }
    (!digits.is_empty()).then_some(digits)
}

fn integer_from_text(text: &str) -> Option<Value> {
    let text = text.trim();
    let (whole, fraction) = match text.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (text, None),
    };
    if fraction.is_some_and(|zeros| zeros.is_empty() || zeros.bytes().any(|byte| byte != b'0')) {
        return None;
    }
    let (negative, unsigned) = match whole.as_bytes().first() {
        Some(b'-') => (true, &whole[1..]),
        Some(b'+') => (false, &whole[1..]),
        _ => (false, whole),
    };
    let digits = without_digit_separators(unsigned)?;
    if negative {
        format!("-{digits}").parse::<i64>().ok().map(Value::from)
    } else {
        digits.parse::<u64>().ok().map(Value::from)
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the round trip back to f64 rejects saturated or inexact casts"
)]
fn integer_from_float(number: f64) -> Option<Value> {
    if number.fract() != 0.0 {
        return None;
    }
    if number >= 0.0 {
        let integer = number as u64;
        (integer as f64 == number && integer != u64::MAX).then(|| Value::from(integer))
    } else {
        let integer = number as i64;
        (integer as f64 == number).then(|| Value::from(integer))
    }
}

fn float_from_text(text: &str) -> Option<Value> {
    let text = text.trim();
    for run in text
        .split(['.', 'e', 'E', '+', '-'])
        .filter(|run| !run.is_empty())
    {
        without_digit_separators(run)?;
    }
    text.replace('_', "")
        .parse::<f64>()
        .ok()
        .and_then(Number::from_f64)
        .map(Value::Number)
}

fn boolean_from_text(text: &str) -> Option<Value> {
    match text.to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Some(Value::Bool(true)),
        "0" | "off" | "f" | "false" | "n" | "no" => Some(Value::Bool(false)),
        _ => None,
    }
}

fn coerce_primitive(value: &Value, kind: &str) -> Option<Value> {
    match (kind, value) {
        ("integer", Value::Number(number)) => number.as_f64().and_then(integer_from_float),
        ("integer", Value::String(text)) => integer_from_text(text),
        ("integer", Value::Bool(flag)) => Some(Value::from(i64::from(*flag))),
        ("number", Value::String(text)) => float_from_text(text),
        ("number", Value::Bool(flag)) => {
            Number::from_f64(f64::from(u8::from(*flag))).map(Value::Number)
        }
        ("boolean", Value::Number(number)) => match number.as_f64() {
            Some(1.0) => Some(Value::Bool(true)),
            Some(0.0) => Some(Value::Bool(false)),
            _ => None,
        },
        ("boolean", Value::String(text)) => boolean_from_text(text),
        _ => None,
    }
}

fn coerce_union(value: Value, members: &[Value]) -> Value {
    if members.iter().any(|member| satisfies(&value, member)) {
        return value;
    }
    for member in members {
        let coerced = coerce_with_schema(value.clone(), member);
        if satisfies(&coerced, member) {
            return coerced;
        }
    }
    value
}

fn coerce_with_schema(value: Value, schema: &Value) -> Value {
    let mut value = value;
    if let Some(Value::Array(members)) = schema.get("allOf") {
        for member in members {
            value = coerce_with_schema(value, member);
        }
    }
    if let Some(Value::Array(members)) = schema.get("anyOf") {
        value = coerce_union(value, members);
    }
    if let Some(Value::Array(members)) = schema.get("oneOf") {
        value = coerce_union(value, members);
    }
    let kinds = schema_types(schema);
    if !kinds.is_empty() && !kinds.iter().any(|kind| matches_type(&value, kind)) {
        if let Some(coerced) = kinds.iter().find_map(|kind| coerce_primitive(&value, kind)) {
            value = coerced;
        }
    }
    if kinds.contains(&"object") {
        if let Value::Object(object) = &mut value {
            coerce_object(object, schema);
        }
    }
    if kinds.contains(&"array") {
        if let Value::Array(items) = &mut value {
            coerce_array(items, schema);
        }
    }
    value
}

fn coerce_object(object: &mut Map<String, Value>, schema: &Value) {
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, property) in properties {
            if let Some(field) = object.get_mut(name) {
                *field = coerce_with_schema(field.take(), property);
            }
        }
    }
    if let Some(additional @ Value::Object(_)) = schema.get("additionalProperties") {
        for (name, field) in object.iter_mut() {
            if properties.is_some_and(|properties| properties.contains_key(name)) {
                continue;
            }
            *field = coerce_with_schema(field.take(), additional);
        }
    }
}

fn coerce_array(items: &mut [Value], schema: &Value) {
    match schema.get("items") {
        Some(Value::Array(item_schemas)) => {
            for (item, item_schema) in items.iter_mut().zip(item_schemas) {
                *item = coerce_with_schema(item.take(), item_schema);
            }
        }
        Some(item_schema @ Value::Object(_)) => {
            for item in items.iter_mut() {
                *item = coerce_with_schema(item.take(), item_schema);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "argument_coercion_test.rs"]
mod tests;
