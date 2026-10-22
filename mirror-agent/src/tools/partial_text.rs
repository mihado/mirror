//! Text of a known length of which only a prefix may be held.

/// Text of a known length of which only a prefix may be held.
pub(crate) struct PartialText {
    /// The text, or a prefix of it at least as long as the character cap.
    pub(crate) kept: String,
    /// The whole text's length in characters.
    pub(crate) chars: usize,
    /// The whole text's line breaks.
    pub(crate) newlines: usize,
}

impl PartialText {
    /// Text held whole.
    pub(crate) fn whole(text: String) -> Self {
        Self {
            chars: text.chars().count(),
            newlines: text.matches('\n').count(),
            kept: text,
        }
    }

    /// Whether `kept` is the whole text.
    pub(crate) fn is_whole(&self) -> bool {
        self.kept.chars().count() == self.chars
    }

    /// This text followed by `next`, held for as long as this one is whole.
    pub(crate) fn concat(mut self, next: Self) -> Self {
        if self.is_whole() {
            self.kept.push_str(&next.kept);
        }
        Self {
            kept: self.kept,
            chars: self.chars + next.chars,
            newlines: self.newlines + next.newlines,
        }
    }
}
