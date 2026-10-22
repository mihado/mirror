"""Bound disposable prose; canonical source remains available for reading.

Ordinary newlines are safe, but block structure keeps the canonical path.
"""

import re

_CONTAINER = re.compile(r'(?:[-+*]|[0-9]{1,9}[.)])\s')


class ProsePreview:
  """Scan additions once and render at most 8 KiB of eligible live prose."""

  def __init__(self) -> None:
    self._source = ''
    self._line_start = 0
    self._preview_start = 0
    self._blocked = False
    self._link = _InlineLinks()

  def update(self, source: str) -> tuple[str, int]:
    if not source.startswith(self._source):
      self._line_start = 0
      self._preview_start = 0
      self._blocked = False
      self._link = _InlineLinks()
      self._source = ''
    start = len(self._source)
    added = source[start:]
    # Keep earlier fenced or nested block context; cropping changes its meaning.
    self._blocked |= any(character in added for character in '\r\x00\x1b|')
    for match in re.finditer('\n', added):
      self._blocked |= _structural(source, self._line_start)
      self._line_start = start + match.end()
    self._source = source
    if self._blocked:
      return source, 0
    self._link.scan(source)
    if _structural(source, self._line_start):
      return source, 0
    # A destination grows invisibly until it closes; the label stays visible.
    end = self._link.destination_start
    if end is None:
      end = len(source)
    tail = source[max(self._preview_start, end - 8192) : end].encode('utf-8')
    if end - self._preview_start > 8192 or len(tail) > 8192:
      # Leave room to grow without discarding stable Markdown on every delta.
      tail = tail[-7168:]
    content = tail.decode('utf-8', errors='ignore')
    self._preview_start = end - len(content)
    return content, self._preview_start


def _structural(source: str, start: int) -> bool:
  """Leave fences, containers, HTML, tables and math to the block renderer."""
  return (
    source.startswith((' ', '\t', '>', '<', '```', '~~~', '$$', '\\['), start)
    or len(source) - start <= 2
    and source[start:] in ('`', '``', '~', '~~')
    or _CONTAINER.match(source, start) is not None
  )


class _InlineLinks:
  """Lex only inline punctuation, retaining the first unfinished destination."""

  _MARKS = re.compile(r'\\[\s\S]?|`+|[][()]')
  _URL_MARKS = re.compile(r"\\[\s\S]?|[()<>\"']")
  _TICKS = re.compile(r'`+')

  def __init__(self) -> None:
    self.destination_start: int | None = None
    self._cursor = 0
    self._label_depth = 0
    self._label_end = -1
    self._code_width = 0
    self._parentheses = 0
    self._quoted = ''

  def scan(self, source: str) -> None:
    """Resume tokenization at the last complete escape or delimiter run."""
    while self._cursor < len(source):
      pattern = self._MARKS
      if self.destination_start is not None:
        pattern = self._URL_MARKS
      elif self._code_width:
        pattern = self._TICKS
      match = pattern.search(source, self._cursor)
      if match is None:
        self._cursor = len(source)
        return
      token = match[0]
      if match.end() == len(source) and (token == '\\' or token[0] == '`'):
        # A following chunk may complete the escape or extend the backtick run.
        self._cursor = match.start()
        return
      self._cursor = match.end()
      if token[0] == '\\':
        continue
      if self.destination_start is not None:
        self._destination(source, match.start(), token, self.destination_start)
      elif token[0] == '`':
        if not self._code_width:
          self._code_width = len(token)
        elif self._code_width == len(token):
          self._code_width = 0
      elif token == '[':
        self._label_depth += 1
      elif token == ']' and self._label_depth:
        self._label_depth -= 1
        self._label_end = match.end()
      elif token == '(' and match.start() == self._label_end:
        self.destination_start = match.start()
        self._parentheses = 1

  def _destination(
    self, source: str, index: int, token: str, start: int
  ) -> None:
    if self._quoted:
      if token == self._quoted:
        self._quoted = ''
    elif self._parentheses == 1 and (
      token == '<'
      and (index == start + 1 or source[index - 1].isspace())
      or token in ('"', "'")
      and source[index - 1].isspace()
    ):
      self._quoted = '>' if token == '<' else token
    elif token == '(':
      self._parentheses += 1
    elif token == ')':
      self._parentheses -= 1
      if not self._parentheses:
        self.destination_start = None
