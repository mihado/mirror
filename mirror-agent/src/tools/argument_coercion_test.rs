use pretty_assertions::assert_eq;
use serde_json::{json, Value};

use super::coerce_to_schema;

fn coerced(mut value: Value, schema: Value) -> Value {
    coerce_to_schema(&mut value, &schema);
    value
}

fn object_of(kind: Value, names: &[&str]) -> Value {
    let properties: serde_json::Map<String, Value> = names
        .iter()
        .map(|name| ((*name).to_string(), json!({"type": kind})))
        .collect();
    json!({"type": "object", "properties": properties})
}

#[test]
fn integers_parse_as_pydantic_does() {
    assert_eq!(
        coerced(
            json!({"a": "10.0", "b": " +1_000 ", "c": 10.0, "d": true, "e": "9007199254740993", "f": "-0", "g": 9_223_372_036_854_775_808.0, "h": -9_223_372_036_854_775_808.0}),
            object_of(
                json!(["integer", "null"]),
                &["a", "b", "c", "d", "e", "f", "g", "h"]
            ),
        ),
        json!({"a": 10, "b": 1000, "c": 10, "d": 1, "e": 9_007_199_254_740_993_u64, "f": 0, "g": 9_223_372_036_854_775_808_u64, "h": i64::MIN}),
    );
}

#[test]
fn integers_pydantic_rejects_stay_as_sent() {
    let sent = json!({"a": "1e1", "b": "1.5", "c": "10.", "d": "1__0", "e": 1.5, "f": ""});
    assert_eq!(
        coerced(
            sent.clone(),
            object_of(json!("integer"), &["a", "b", "c", "d", "e", "f"])
        ),
        sent,
    );
}

#[test]
fn numbers_parse_as_pydantic_does() {
    assert_eq!(
        coerced(
            json!({"a": "1e1", "b": " 0.5 ", "c": true, "d": "1_000.5", "e": "inf"}),
            object_of(json!("number"), &["a", "b", "c", "d", "e"]),
        ),
        json!({"a": 10.0, "b": 0.5, "c": 1.0, "d": 1000.5, "e": "inf"}),
    );
}

#[test]
fn booleans_parse_as_pydantic_does() {
    assert_eq!(
        coerced(
            json!({"a": "TRUE", "b": "yes", "c": "off", "d": 1, "e": 0.0, "f": " true ", "g": 2, "h": "1.0"}),
            object_of(json!("boolean"), &["a", "b", "c", "d", "e", "f", "g", "h"]),
        ),
        json!({"a": true, "b": true, "c": false, "d": true, "e": false, "f": " true ", "g": 2, "h": "1.0"}),
    );
}

#[test]
fn strings_and_nulls_are_never_coerced() {
    let schema = json!({
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "label": {"type": "string"},
            "flag": {"type": "boolean"},
            "count": {"type": "integer"},
        },
    });
    let sent = json!({"pattern": 7, "label": true, "flag": null, "count": null});
    assert_eq!(coerced(sent.clone(), schema), sent);
}

#[test]
fn nested_objects_arrays_and_unions_are_coerced() {
    let schema = json!({
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {"type": "object", "properties": {"line": {"type": "integer"}, "text": {"type": "string"}}},
            },
            "target": {"anyOf": [{"type": "integer"}, {"type": "array", "items": {"type": "integer"}}]},
        },
    });
    assert_eq!(
        coerced(
            json!({"edits": [{"line": "3", "text": "x"}], "target": "4"}),
            schema
        ),
        json!({"edits": [{"line": 3, "text": "x"}], "target": 4}),
    );
}

#[test]
fn union_members_that_already_match_are_left_alone() {
    let schema = json!({"anyOf": [{"type": "string"}, {"type": "integer"}]});
    assert_eq!(coerced(json!("15"), schema), json!("15"));
}
