"""Retained syntax rows for the Python frontend's streaming code fences."""

from __future__ import annotations

import collections
import re
import sys
from collections.abc import Callable, Iterable
from typing import override

from pygments import lexer as _lexer
from pygments import token as _token
import pygments.lexers.python
from rich import console as _console
from rich import segment as _segment
from rich import style as _style
from rich import syntax as _syntax
from rich import text as _text


def _plain(code: str) -> bool:
  """Bound highlighting work and fall back to plain text for oversized code."""
  return (
    len(code.encode('utf-8')) > 512 * 1024
    or '\0' in code
    or code.count('\n') + bool(code and not code.endswith('\n')) > 10_000
    or any(len(line.encode('utf-8')) > 4096 for line in code.split('\n'))
  )


class _PythonLexer(_lexer.ExtendedRegexLexer):
  """Use the existing Python grammar with Pygments' explicit context API."""

  tokens = pygments.lexers.python.PythonLexer.tokens
  flags = pygments.lexers.python.PythonLexer.flags


class _PythonParser:
  """Checkpoint complete lines; keep lookahead-sensitive constructs mutable."""

  def __init__(
    self,
    theme: _syntax.SyntaxTheme,
    style: _style.Style,
    tab_size: int,
    background: str | None,
  ) -> None:
    self._lexer = _PythonLexer()
    self._theme = theme
    self._style = style
    self._tab_size = tab_size
    self._background = background
    self._stack = ['root']

  def parse(self, code: str, start: int) -> tuple[list[_text.Text], int]:
    # Preserve the full source for anchors/lookbehind, including the first line.
    source = code if code.endswith('\n') else code + '\n'
    context = _lexer.LexerContext(source, start, self._stack.copy())
    rendered = _text.Text(
      style=self._style,
      tab_size=self._tab_size,
      no_wrap=False,
      justify='default' if self._style.transparent_background else 'left',
    )
    committed, stack = start, self._stack
    mutable = False
    interpolation = False
    for position, kind, value in self._lexer.get_tokens_unprocessed(
      context=context
    ):
      line_start = source.rfind('\n', 0, position) + 1
      line_end = source.find('\n', position)
      # Delegate callbacks have submatch positions, not resumable checkpoints.
      if (
        position == line_start == context.pos
        and 'soft-keywords-inner' not in context.stack
        and line_end < len(code)
        and source[position:line_end].strip()
      ):
        if context.stack == ['root']:
          mutable = False
        if not mutable and not interpolation:
          committed, stack = position, context.stack.copy()
      # A closing docstring can restyle its opening and every intervening line.
      if (
        context.stack == ['root']
        and value in ('"""', "'''")
        and re.fullmatch(r'\s*[rRuUbB]{0,2}', source[line_start:position])
      ):
        mutable = True
      # Interpolations can also span lines and acquire their style on closure.
      if (
        context.stack != ['root']
        and '{' in value
        and kind is not _token.String.Escape
      ):
        interpolation = True
      if kind is _token.String.Interpol and '}' in value:
        interpolation = False
      # ExtendedRegexLexer uses Text for newline recovery; Python uses Whitespace.
      if kind is _token.Text and value == '\n':
        kind = _token.Whitespace
      rendered.append(value, self._theme.get_style_for_token(kind))
    self._stack = stack
    if self._background is not None:
      rendered.stylize(f'on {self._background}')
    return list(rendered.split('\n')), committed


class _HighlightedCode:
  """Share the committed styled prefix across syntax objects and viewport sizes."""

  def __init__(self, syntax: Syntax) -> None:
    self.key = syntax.highlight_key
    self._parser = syntax.parser()
    self.source = ''
    self._lines: list[_text.Text] = []

  def append(self, code: str) -> list[_text.Text] | None:
    if self._parser is None:
      return None
    lines, committed = self._parser.parse(code, len(self.source))
    count = code.count('\n', len(self.source), committed)
    self._lines.extend(lines[:count])
    self.source = code[:committed]
    return [*self._lines, *lines[count:]]


type RowKey = tuple[
  str, tuple[_text.Span, ...], str | _style.Style, int, _style.Style
]


class RowCache:
  """Retain recent wrapped lines within an 8 MiB budget per streaming answer."""

  def __init__(self) -> None:
    self._rows: collections.OrderedDict[
      RowKey, tuple[tuple[_segment.Segment, ...], int]
    ] = collections.OrderedDict()
    self._bytes = 0
    self._highlight: _HighlightedCode | None = None

  def get(
    self,
    key: RowKey,
    render: Callable[[], tuple[_segment.Segment, ...]],
  ) -> tuple[_segment.Segment, ...]:
    cached = self._rows.get(key)
    if cached is not None:
      self._rows.move_to_end(key)
      return cached[0]
    row = render()
    size = (
      sys.getsizeof(key) + sum(map(sys.getsizeof, key)) + sys.getsizeof(row)
    )
    size += sum(
      sys.getsizeof(span) + sys.getsizeof(span.style) for span in key[1]
    )
    size += sum(
      sys.getsizeof(segment)
      + sys.getsizeof(segment.text)
      + sys.getsizeof(segment.style)
      for segment in row
    )
    if size <= 8 * 1024 * 1024:
      while self._rows and self._bytes + size > 8 * 1024 * 1024:
        _, (_, removed) = self._rows.popitem(last=False)
        self._bytes -= removed
      self._rows[key] = (row, size)
      self._bytes += size
    return row

  def highlight(self, syntax: Syntax, code: str) -> list[_text.Text] | None:
    # Match the canonical lexer's input normalization before checking the prefix.
    code = code.replace('\r\n', '\n').replace('\r', '\n').removeprefix('\ufeff')
    key = syntax.highlight_key
    if (
      self._highlight is None
      or self._highlight.key != key
      or not code.startswith(self._highlight.source)
    ):
      self._highlight = _HighlightedCode(syntax)
    return self._highlight.append(code)


class Syntax(_syntax.Syntax):
  """Reuse wrapped rows while keeping the existing Python syntax highlighter."""

  def __init__(
    self,
    code: str,
    lexer: str,
    *,
    rows: RowCache | None,
    theme: str,
    background_color: str,
    padding: int | tuple[int, int],
  ) -> None:
    super().__init__(
      code,
      lexer,
      theme=theme,
      word_wrap=True,
      background_color=background_color,
      padding=padding,
    )
    self._rows = rows
    self._language = lexer
    self._theme_name = theme

  @property
  def highlight_key(self) -> tuple[str, str, _style.Style, int]:
    return self._language, self._theme_name, self.base_style, self.tab_size

  def parser(self) -> _PythonParser | None:
    # Other lexers may use custom callbacks, lookahead or additional state.
    if type(self.lexer) is pygments.lexers.python.PythonLexer:
      return _PythonParser(
        self._theme, self.base_style, self.tab_size, self.background_color
      )
    return None

  @override
  def highlight(
    self,
    code: str,
    line_range: tuple[int | None, int | None] | None = None,
  ) -> _text.Text:
    if _plain(code):
      return _text.Text(
        code,
        style=self._get_base_style(),
        no_wrap=False,
        tab_size=self.tab_size,
      )
    return super().highlight(code, line_range)

  def _highlight_lines(self, code: str) -> list[_text.Text]:
    if self._rows is not None and not _plain(code):
      lines = self._rows.highlight(self, code)
      if lines is not None:
        return lines
    return list(self.highlight(code).split('\n'))

  @property
  def plain(self) -> bool:
    return _plain(self.code.expandtabs(self.tab_size))

  @property
  def base_style(self) -> _style.Style:
    return self._get_base_style()

  def lines(self) -> list[_text.Text]:
    """Return styled source lines before terminal wrapping."""
    return self._highlight_lines(self.code.expandtabs(self.tab_size))

  @property
  def retained(self) -> bool:
    return self._rows is not None

  @override
  def __rich_console__(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> _console.RenderResult:
    if self._rows is None:
      yield from super().__rich_console__(console, options)
      return
    background = self._get_base_style()
    left = _segment.Segment(' ' * self.padding[3], background)
    right = _segment.Segment(' ' * self.padding[1], background)
    for row in _segment.Segment.split_lines(self._get_syntax(console, options)):
      yield left
      yield from row
      yield right
      yield _segment.Segment.line()

  @override
  def _get_syntax(
    self, console: _console.Console, options: _console.ConsoleOptions
  ) -> Iterable[_segment.Segment]:
    if self._rows is None:
      yield from super()._get_syntax(console, options)
      return
    width = max(0, options.max_width - self.padding[1] - self.padding[3])
    lines = self._highlight_lines(self.code.expandtabs(self.tab_size))
    background = self._get_base_style()
    render_options = options.update(width=width, height=None, justify='left')
    for line in lines:
      yield from self._render_line(
        console, render_options, background, width, line
      )

  def _render_line(
    self,
    console: _console.Console,
    render_options: _console.ConsoleOptions,
    background: _style.Style,
    width: int,
    line: _text.Text,
  ) -> Iterable[_segment.Segment]:
    assert self._rows is not None

    def render() -> tuple[_segment.Segment, ...]:
      return tuple(
        segment
        for row in console.render_lines(
          line,
          render_options,
          style=background,
          pad=True,
        )
        for segment in (*row, _segment.Segment.line())
      )

    # A single generated line must not retain an unbounded source string.
    if len(line.plain) > 8192:
      yield from render()
    else:
      key = (line.plain, tuple(line.spans), line.style, width, background)
      yield from self._rows.get(key, render)
