"""Retain final viewport rows for the complete lines of a live code fence."""

from __future__ import annotations

import sys
from collections.abc import Iterable

from prompt_toolkit import formatted_text
from rich import color, console, segment, text

from mirror.cli import code_rendering, hyperlinks, rendering, themes
from mirror.cli.composer import skill_completion

type Row = tuple[formatted_text.OneStyleAndTextTuple, ...]
type Rows = tuple[Row, ...]


def _size(row: Row) -> int:
  return sys.getsizeof(row) + sum(
    sys.getsizeof(fragment)
    + sys.getsizeof(fragment[0])
    + sys.getsizeof(fragment[1])
    for fragment in row
  )


class CodeLayout:
  """Retain one live fence; edits and presentation changes rebuild its rows."""

  def __init__(self) -> None:
    self._source = ''
    self._key: tuple[object, ...] | None = None
    self._highlighted = code_rendering.RowCache()
    self._line_count = 0
    self._lines: list[text.Text] = []
    self._complete: Rows = ()
    self._complete_bytes = 0
    self._rows: Rows = ()
    self._bytes = 0

  def render(
    self,
    answer: rendering.Answer,
    painter: console.Console,
    width: int,
    padding: int,
    mono: bool,
  ) -> tuple[Rows, int] | None:
    markdown = answer.markdown
    if (
      markdown is None
      or len(markdown.parsed) != 1
      or markdown.parsed[0].type != 'fence'
      or width < 11
    ):
      return None
    token = markdown.parsed[0]
    code = token.content.rstrip()
    root = painter.get_style('ui.text')
    rail = painter.get_style('ui.code-rail')
    background = painter.get_style('ui.code')
    syntax_theme = themes.syntax_theme(painter, 'code')
    key = (
      width,
      padding,
      mono,
      root,
      rail,
      background,
      syntax_theme,
      token.info,
      answer.continuation,
      bool(token.content),
      answer.skill_names,
      painter.get_style('ui.accent'),
      painter.get_style('ui.info'),
    )
    if key == self._key and code == self._source:
      return self._rows, self._bytes
    syntax = code_rendering.Syntax(
      code,
      token.info.partition(' ')[0] or 'text',
      rows=self._highlighted,
      theme=syntax_theme,
      background_color=background.bgcolor.name
      if background.bgcolor
      else 'default',
      padding=0,
    )
    if syntax.plain:
      return None
    highlighted = syntax.lines()
    if syntax.plain:
      return None
    # Python lexers may restyle earlier lines when more source arrives.
    reuse = (
      key == self._key
      and code.startswith(self._source)
      and highlighted[: self._line_count] == self._lines[: self._line_count]
    )
    complete = list(self._complete) if reuse else []
    complete_bytes = self._complete_bytes if reuse else 0
    lines = highlighted[self._line_count if reuse else 0 :]
    body: Rows = ()
    plain_gutter = segment.Segment(' ' * 3)
    rail_segment = segment.Segment('┃', rail)
    syntax_background = syntax.base_style
    side = segment.Segment(' ', syntax_background)
    system = (
      color.ColorSystem.STANDARD
      if painter.color_system == 'standard'
      else color.ColorSystem.TRUECOLOR
    )

    def convert(segments: Iterable[segment.Segment]) -> Row:
      decorated = skill_completion.highlight_renderable(
        segment.Segments(segments), answer.skill_names
      )
      result: formatted_text.StyleAndTextTuples = []
      for value in painter.render(decorated):
        appearance = root + value.style
        result.extend(
          hyperlinks.from_rich(
            value.text, appearance.without_color if mono else appearance, system
          )
        )
      return tuple(result)

    for index, line in enumerate(lines):
      wrapped = painter.render_lines(
        line,
        # Three gutter cells, one rail and one inset on each side.
        painter.options.update(width=width - 6, height=None, justify='left'),
        style=syntax_background,
        pad=True,
      )
      rendered = tuple(
        convert((plain_gutter, rail_segment, side, *row, side))
        for row in wrapped
      )
      if index < len(lines) - 1:
        complete.extend(rendered)
        complete_bytes += sum(map(_size, rendered))
      else:
        body = rendered
    blank = segment.Segment(' ' * (width - 4), background)
    header = (
      convert((
        segment.Segment('● ', painter.get_style('ui.accent')),
        segment.Segment(' '),
        rail_segment,
        blank,
      ))
      if not answer.continuation
      else convert((plain_gutter, rail_segment, blank))
    )
    empty = convert((plain_gutter, rail_segment, blank))
    footer = list(empty)
    while footer and footer[-1][0] == '':
      item = footer.pop()
      value = item[1].rstrip()
      if value:
        footer.append(('', value))
        break
    rows = (
      (header,)
      + (empty,) * (1 if token.content else 0)
      + tuple(complete)
      + body
      + (tuple(footer),)
      + ((),) * padding
    )
    size = (
      complete_bytes
      + sum(map(_size, (header, empty, *body)))
      + sys.getsizeof(rows)
    )
    if size <= 8 * 1024 * 1024:
      self._source, self._key = code, key
      self._line_count = max(0, len(highlighted) - 1)
      self._lines = highlighted
      self._complete, self._complete_bytes = tuple(complete), complete_bytes
      self._rows, self._bytes = rows, size
    return rows, size
