"""Isolate prompt-toolkit optimizations behind TranscriptWindow and install."""

from __future__ import annotations

import dataclasses
import functools
from typing import override

from prompt_toolkit import data_structures, mouse_events
from prompt_toolkit.application import current
from prompt_toolkit.layout import containers, controls
from prompt_toolkit.layout import mouse_handlers as _mouse_handlers
from prompt_toolkit.layout import screen as _screen

from mirror.cli import transcript_view

type _Row = tuple[tuple[str, str], ...]
type _Cells = tuple[dict[int, _screen.Char], dict[int, str]]


@dataclasses.dataclass(frozen=True)
class ScrollRegion:
  """Painted transcript rows and their stable source identities."""

  offset: int
  rows: tuple[_Row, ...]
  positions: tuple[transcript_view.RowPosition, ...]


class _PreparedRows(controls.UIControl):
  def __init__(self) -> None:
    self.content = controls.UIContent()

  @override
  def create_content(self, width: int, height: int) -> controls.UIContent:
    del width, height
    return self.content


class TranscriptWindow(containers.Window):
  """A read-only viewport caching cell conversion only for the last painted rows."""

  scroll_region: ScrollRegion | None = None

  def __init__(self, view: transcript_view.TranscriptView) -> None:
    self._view = view
    self._cache: dict[_Row, _Cells] = {}
    self._shape: tuple[int, str, bool] | None = None
    super().__init__(view, wrap_lines=False, always_hide_cursor=True)
    self._rows = _PreparedRows()
    self._painter = containers.Window(
      self._rows, wrap_lines=False, always_hide_cursor=True
    )

  @override
  def reset(self) -> None:
    self.scroll_region = None
    self._cache.clear()
    super().reset()

  def _prepare_rows(
    self, rows: list[_Row], width: int, parent_style: str, erase_bg: bool
  ) -> None:
    shape = (width, parent_style, erase_bg)
    if shape != self._shape:
      self._cache.clear()
      self._shape = shape
    missing = list(dict.fromkeys(row for row in rows if row not in self._cache))
    if missing:
      self._rows.content = controls.UIContent(
        get_line=lambda index: list(missing[index]),
        line_count=len(missing),
        show_cursor=False,
      )
      prepared = _screen.Screen()
      # An extra row keeps Window's positional last-line style out of the cache.
      self._painter.write_to_screen(
        prepared,
        _mouse_handlers.MouseHandlers(),
        _screen.WritePosition(0, 0, width, len(missing) + 1),
        parent_style,
        erase_bg,
        None,
      )
      for index, row in enumerate(missing):
        self._cache[row] = (
          prepared.data_buffer[index],
          prepared.zero_width_escapes[index],
        )
      self._rows.content = controls.UIContent()
      self._painter.render_info = None
    self._cache = {row: self._cache[row] for row in rows}

  @override
  def write_to_screen(
    self,
    screen: _screen.Screen,
    mouse_handlers: _mouse_handlers.MouseHandlers,
    write_position: _screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    self.scroll_region = None
    if not write_position.width or not write_position.height:
      return
    # Stock Window owns overlay composition; cache the unoccluded transcript.
    if (
      write_position.xpos != 0
      or (z_index is not None and z_index > 0)
      or any(
        screen.data_buffer.get(y)
        for y in range(
          write_position.ypos, write_position.ypos + write_position.height
        )
      )
    ):
      self._cache.clear()
      super().write_to_screen(
        screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
      )
      return
    content = self.content.create_content(
      write_position.width, write_position.height
    )
    rows = [
      tuple((style, text) for style, text, *_ in content.get_line(index))
      if index < content.line_count
      else ()
      for index in range(write_position.height)
    ]
    movable = self._view.scrolling_range()
    if movable:
      self.scroll_region = ScrollRegion(
        movable.start,
        tuple(rows[movable.start : movable.stop]),
        self._view.scrolling_positions(),
      )
    self._prepare_rows(rows, write_position.width, parent_style, erase_bg)
    for index, row in enumerate(rows):
      cells, escapes = self._cache[row]
      y = write_position.ypos + index
      # Copy the mappings: selection and floats can replace cells in this frame.
      screen.data_buffer[y].update(cells)
      screen.zero_width_escapes[y].update(escapes)
    screen.fill_area(
      _screen.WritePosition(
        write_position.xpos,
        write_position.ypos + write_position.height - 1,
        write_position.width,
        1,
      ),
      'class:last-line',
      after=True,
    )
    screen.height = max(
      screen.height, write_position.ypos + write_position.height
    )
    screen.visible_windows_to_write_positions[self] = write_position
    self.render_info = containers.WindowRenderInfo(
      window=self,
      ui_content=content,
      horizontal_scroll=0,
      vertical_scroll=0,
      window_width=write_position.width,
      window_height=write_position.height,
      configured_scroll_offsets=self.scroll_offsets,
      visible_line_to_row_col={
        i: (i, 0) for i in range(min(content.line_count, write_position.height))
      },
      rowcol_to_yx={},
      x_offset=write_position.xpos,
      y_offset=write_position.ypos,
      wrap_lines=False,
    )
    mouse_handlers.set_mouse_handler_for_range(
      write_position.xpos,
      write_position.xpos + write_position.width,
      write_position.ypos,
      write_position.ypos + write_position.height,
      functools.partial(self._dispatch_mouse, content, write_position),
    )

  def _dispatch_mouse(
    self,
    content: controls.UIContent,
    position: _screen.WritePosition,
    event: mouse_events.MouseEvent,
  ) -> None:
    if self not in current.get_app().layout.walk_through_modal_area():
      return
    row = min(event.position.y - position.ypos, content.line_count - 1)
    column = -1
    x = source_column = 0
    # No input cursor: map columns only for pointer events, not every painted cell.
    if row >= 0:
      for style, text, *_ in content.get_line(row):
        if '[ZeroWidthEscape]' in style:
          continue
        for character in text:
          if x <= event.position.x - position.xpos and x < position.width:
            column = source_column
          x += _screen.Char(character).width
          source_column += 1
    point = (
      data_structures.Point(column, row)
      if column >= 0
      else data_structures.Point(0, 0)
    )
    self.content.mouse_handler(
      mouse_events.MouseEvent(
        point,
        event.event_type,
        event.button,
        event.modifiers,
      )
    )
