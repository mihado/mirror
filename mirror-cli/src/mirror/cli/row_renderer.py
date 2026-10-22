"""Retained terminal frames, scrolling and mouse regions for prompt-toolkit."""

from __future__ import annotations

import collections
import copy
import operator
import os
from collections.abc import Callable, Mapping
from typing import override

from prompt_toolkit import (
  application,
  data_structures,
  mouse_events,
  renderer,
  styles,
)
from prompt_toolkit.key_binding import key_bindings
from prompt_toolkit.layout import containers, mouse_handlers, screen
from prompt_toolkit.layout import layout as _layout
from prompt_toolkit.output import base as output_base
from prompt_toolkit.output import vt100

from mirror.cli import transcript_window

type _Row = tuple[tuple[str, str], ...]
type _Shape = tuple[
  transcript_window.TranscriptWindow, int, int, data_structures.Size
]


type _MouseRegion = tuple[int, int, int, int, mouse_handlers.MouseHandler]


def _unhandled(
  event: mouse_events.MouseEvent,
) -> key_bindings.NotImplementedOrNone:
  del event
  return NotImplemented


class _MouseRow(collections.defaultdict[int, mouse_handlers.MouseHandler]):
  def __init__(self, regions: list[_MouseRegion], y: int) -> None:
    super().__init__()
    self._regions, self._y = regions, y

  @override
  def __missing__(self, x: int) -> mouse_handlers.MouseHandler:
    for left, right, top, bottom, handler in reversed(self._regions):
      if left <= x < right and top <= self._y < bottom:
        return handler
    return _unhandled


class _Rows(
  collections.defaultdict[
    int, collections.defaultdict[int, mouse_handlers.MouseHandler]
  ]
):
  def __init__(self, regions: list[_MouseRegion]) -> None:
    super().__init__()
    self._regions = regions

  @override
  def __missing__(
    self, y: int
  ) -> collections.defaultdict[int, mouse_handlers.MouseHandler]:
    row = _MouseRow(self._regions, y)
    self[y] = row
    return row


class MouseHandlers(mouse_handlers.MouseHandlers):
  def __init__(self) -> None:
    self._regions: list[_MouseRegion] = []
    self.mouse_handlers = _Rows(self._regions)

  @override
  def set_mouse_handler_for_range(
    self,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    handler: mouse_handlers.MouseHandler,
  ) -> None:
    if x_min >= x_max or y_min >= y_max:
      return
    self._regions.append((x_min, x_max, y_min, y_max, handler))
    # Rows touched by third-party containers can already contain direct writes.
    # Preserve the same last-registration precedence as the stock cell grid.
    for y, row in self.mouse_handlers.items():
      if y_min <= y < y_max:
        for x in row:
          if x_min <= x < x_max:
            row[x] = handler

  def copy_to(
    self,
    target: mouse_handlers.MouseHandlers,
    position: screen.WritePosition,
    wrap: Callable[[mouse_handlers.MouseHandler], mouse_handlers.MouseHandler],
  ) -> None:
    left, right = position.xpos, position.xpos + position.width
    top, bottom = position.ypos, position.ypos + position.height
    for x1, x2, y1, y2, handler in self._regions:
      x1, x2 = max(left, x1), min(right, x2)
      y1, y2 = max(top, y1), min(bottom, y2)
      if x1 < x2 and y1 < y2:
        target.set_mouse_handler_for_range(x1, x2, y1, y2, wrap(handler))
    # Normally empty: only legacy containers write individual cells.
    for y, row in self.mouse_handlers.items():
      if top <= y < bottom:
        for x, handler in row.items():
          if left <= x < right:
            target.mouse_handlers[y][x] = wrap(handler)


def _line_end(row: dict[int, screen.Char], styled: Mapping[str, bool]) -> int:
  return max(
    (x for x, cell in row.items() if cell.char != ' ' or styled[cell.style]),
    default=0,
  )


def _move(
  output: output_base.Output,
  before: data_structures.Point,
  after: data_structures.Point,
  width: int,
) -> None:
  # Full-screen rows already exist; relative moves also support non-VT outputs.
  if after.y > before.y:
    output.cursor_down(after.y - before.y)
  elif after.y < before.y:
    output.cursor_up(before.y - after.y)
  if before.x >= width - 1:
    output.write('\r')
    output.cursor_forward(after.x)
  elif after.x > before.x:
    output.cursor_forward(after.x - before.x)
  elif after.x < before.x:
    output.cursor_backward(before.x - after.x)


class _FrameBuffer(screen.Screen):
  """Reset a back buffer while retaining its screen and row-map objects.

  Clear the old front buffer before drawing into it again.
  """

  def clear(self) -> None:
    for row in self.data_buffer.values():
      row.clear()
    for row in self.zero_width_escapes.values():
      row.clear()
    self.cursor_positions.clear()
    self.menu_positions.clear()
    self.visible_windows_to_write_positions.clear()
    self._draw_float_functions.clear()
    self.width = self.height = 0
    self.show_cursor = False


class Renderer(renderer.Renderer):
  """Use the normal renderer for lifecycle changes and diff stable frames by row."""

  _spare: _FrameBuffer | None = None
  _previous_region: tuple[_Shape, transcript_window.ScrollRegion] | None = None

  @override
  def reset(
    self, _scroll: bool = False, leave_alternate_screen: bool = True
  ) -> None:
    self._spare = None
    self._previous_region = None
    super().reset(_scroll, leave_alternate_screen)

  @override
  def render[T](
    self,
    app: application.Application[T],
    layout: _layout.Layout,
    is_done: bool = False,
  ) -> None:
    if (
      self.full_screen
      and not is_done
      and not app.exit_style
      and _supported(self.output)
    ):
      frame = copy.copy(layout)
      frame.container = _Frame(layout.container, self._prepare)
      layout = frame
    else:
      self._previous_region = None
    try:
      self._render(app, layout, is_done)
    except Exception:
      self._last_screen = None
      self._previous_region = None
      raise

  def _render[T](
    self, app: application.Application[T], layout: _layout.Layout, is_done: bool
  ) -> None:
    size = self.output.get_size()
    previous = self._last_screen
    attrs = self._attrs_for_style
    styled = self._style_string_has_style
    if (
      not self.full_screen
      or is_done
      or previous is None
      or self._last_size != size
      or self.style.invalidation_hash() != self._last_style_hash
      or app.style_transformation.invalidation_hash()
      != self._last_transformation_hash
      or app.color_depth != self._last_color_depth
      or self.mouse_support() != self._mouse_support_enabled
      or app.exit_style
      or attrs is None
      or styled is None
    ):
      self._spare = None
      super().render(app, layout, is_done)
      return

    # Keep the actual front buffer: transcript scrolling may create a temporary
    # shifted comparison view while painting. Recycle only after the diff.
    front = previous
    target = self._spare if self._spare is not None else _FrameBuffer()
    self._spare = None
    target.clear()
    handlers = MouseHandlers()
    layout.container.write_to_screen(
      target,
      handlers,
      screen.WritePosition(0, 0, size.columns, size.rows),
      '',
      False,
      None,
    )
    target.draw_all_floats()
    # A transcript frame can move terminal rows and update the previous buffer.
    previous = self._last_screen
    assert previous is not None
    self._diff(app, target, previous, size, attrs, styled)
    self._last_screen, self._last_size = target, size
    self._spare = front if isinstance(front, _FrameBuffer) else None
    self.mouse_handlers = handlers
    shape = app.cursor.get_cursor_shape(app)
    if shape != self._last_cursor_shape:
      self.output.set_cursor_shape(shape)
      self._last_cursor_shape = shape
    self.output.flush()
    app.layout.visible_windows = target.visible_windows

  def _diff[T](
    self,
    app: application.Application[T],
    target: screen.Screen,
    previous: screen.Screen,
    size: data_structures.Size,
    attrs: Mapping[str, styles.Attrs],
    styled: Mapping[str, bool],
  ) -> None:
    output = self.output
    position, last_style = self._cursor_pos, self._last_style
    output.hide_cursor()
    # Wide-glyph continuation cells must not be painted independently.
    # Narrower replacements and shortened rows still need their old cells cleared.
    for y in range(min(max(target.height, previous.height), size.rows)):
      row, old = target.data_buffer[y], previous.data_buffer[y]
      escapes, old_escapes = (
        target.zero_width_escapes[y],
        previous.zero_width_escapes[y],
      )
      if row == old and escapes == old_escapes:
        continue
      end = min(size.columns - 1, _line_end(row, styled))
      old_end = min(size.columns - 1, _line_end(old, styled))
      # OSC links belong to emitted glyphs. Replay their row when its text or
      # escapes change, including a closing escape beyond the last glyph.
      replay = bool(escapes or old_escapes)
      if replay:
        output.write_raw('\x1b]8;;\x1b\\')
      x = 0
      repair_until = 0
      while x <= end:
        cell, before = row[x], old[x]
        width = cell.width or 1
        if (
          replay
          or x < repair_until
          or cell.char != before.char
          or cell.style != before.style
        ):
          if position != data_structures.Point(x, y):
            _move(output, position, data_structures.Point(x, y), size.columns)
          if x in escapes:
            output.write_raw(escapes[x])
          if last_style != cell.style:
            if last_style is None or attrs[last_style] != attrs[cell.style]:
              output.set_attributes(attrs[cell.style], app.color_depth)
            last_style = cell.style
          output.write(cell.char)
          position = data_structures.Point(x + width, y)
          if before.width > width:
            repair_until = max(repair_until, x + before.width)
        x += width
      if replay:
        for column, escape in sorted(escapes.items()):
          if end < column < size.columns:
            if position != data_structures.Point(column, y):
              _move(
                output, position, data_structures.Point(column, y), size.columns
              )
            position = data_structures.Point(column, y)
            output.write_raw(escape)
        output.write_raw('\x1b]8;;\x1b\\')
      # An equal blank tail can still overlap the old wide glyph's leader.
      overlaps = any(x <= end < x + cell.width - 1 for x, cell in old.items())
      if end < old_end or overlaps:
        _move(output, position, data_structures.Point(end + 1, y), size.columns)
        position = data_structures.Point(end + 1, y)
        output.reset_attributes()
        last_style = None
        output.erase_end_of_line()
    cursor = target.get_cursor_position(app.layout.current_window)
    if position != cursor:
      _move(output, position, cursor, size.columns)
    output.reset_attributes()
    if target.show_cursor:
      output.show_cursor()
    self._cursor_pos, self._last_style = cursor, None

  def _prepare(self, painted: screen.Screen) -> None:
    size = self.output.get_size()
    previous_region, self._previous_region = (
      self._previous_region,
      _region(painted, size),
    )
    previous = self._last_screen
    if (
      previous is None
      or previous_region is None
      or self._previous_region is None
      or previous_region[0][:2] != self._previous_region[0][:2]
      or previous_region[0][3] != size
      or self._last_size != size
    ):
      return
    shape, region = self._previous_region
    _, top, bottom, _ = shape
    # A growing footer only invalidates the edge, not the shared transcript rows.
    bottom = min(bottom, previous_region[0][2])
    count = bottom - top
    shift = _shift(previous_region[1], region, count)
    if shift and not _scroll_saves_output(
      previous_region[1].rows[:count], region.rows[:count], shift
    ):
      shift = 0
    if shift:
      previous = _shifted(previous, top, bottom, shift)
      self.output.hide_cursor()
      self.output.reset_attributes()
      self.output.write_raw('\x1b]8;;\x1b\\')
      try:
        self.output.write_raw(f'\x1b[{top + 1};{bottom}r')
        self.output.cursor_goto(top + 1, 1)
        operation = 'L' if shift > 0 else 'M'
        self.output.write_raw(f'\x1b[{abs(shift)}{operation}')
      finally:
        self.output.write_raw('\x1b[r')
        self.output.cursor_goto(1, 1)
      self._cursor_pos = data_structures.Point(0, 0)
      self._last_style = None
    self._last_screen = previous


class _Frame(containers.DynamicContainer):
  def __init__(
    self,
    content: containers.Container,
    prepare: Callable[[screen.Screen], None],
  ) -> None:
    super().__init__(lambda: content)
    self._prepare = prepare

  @override
  def write_to_screen(
    self,
    screen: screen.Screen,
    mouse_handlers: mouse_handlers.MouseHandlers,
    write_position: screen.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    handlers = MouseHandlers()
    super().write_to_screen(
      screen, handlers, write_position, parent_style, erase_bg, z_index
    )
    mouse_handlers.mouse_handlers = handlers.mouse_handlers
    # Finish floats and selection before inspecting or moving any painted rows.
    screen.draw_all_floats()
    self._prepare(screen)


def _supported(output: output_base.Output) -> bool:
  if not isinstance(output, vt100.Vt100_Output) or not output.stdout.isatty():
    return False
  term = output.term or os.environ.get('TERM', '')
  # These terminal families support DECSTBM and IL/DL; unknown outputs stay stock.
  return term.split('-')[0] in {
    'xterm',
    'screen',
    'tmux',
    'rxvt',
    'kitty',
    'foot',
    'alacritty',
    'wezterm',
  }


def _region(
  screen: screen.Screen, size: data_structures.Size
) -> tuple[_Shape, transcript_window.ScrollRegion] | None:
  windows = screen.visible_windows_to_write_positions
  candidates = [
    window
    for window in windows
    if isinstance(window, transcript_window.TranscriptWindow)
  ]
  if len(candidates) != 1:
    return None
  window = candidates[0]
  position, region = windows[window], window.scroll_region
  if region is None:
    return None
  top = position.ypos + region.offset
  bottom = top + len(region.rows)
  if (
    position.xpos != 0
    or position.width != size.columns
    or region.offset < 0
    or not 0 <= top < bottom <= size.rows
    or bottom > position.ypos + position.height
    or any(
      other is not window
      and area.ypos < bottom
      and area.ypos + area.height > top
      for other, area in windows.items()
    )
  ):
    return None
  return (window, top, bottom, size), region


def _shift(
  previous: transcript_window.ScrollRegion,
  current: transcript_window.ScrollRegion,
  count: int,
) -> int:
  """Use retained source positions, then verify that the overlap is unchanged."""
  before, after = previous.positions[:count], current.positions[:count]
  shift = 0
  if before and after:
    if after[0] in before:
      shift = -before.index(after[0])
    elif before[0] in after:
      shift = after.index(before[0])
  old_rows, rows = previous.rows[:count], current.rows[:count]
  if 0 < shift <= count // 2 and rows[shift:] == old_rows[:-shift]:
    return shift
  if -(count // 2) <= shift < 0 and rows[:shift] == old_rows[-shift:]:
    return shift
  # Live appends change source revisions even when painted rows still overlap.
  if old_rows != rows:
    for amount in range(1, count // 2 + 1):
      if rows[amount:] == old_rows[:-amount]:
        return amount
      if rows[:-amount] == old_rows[amount:]:
        return -amount
  return 0


def _shifted(
  previous: screen.Screen, top: int, bottom: int, shift: int
) -> screen.Screen:
  result = copy.copy(previous)
  result.data_buffer = previous.data_buffer.copy()
  result.zero_width_escapes = previous.zero_width_escapes.copy()
  blank = screen.Screen()
  for row in range(top, bottom):
    source = row - shift
    # Stock diff fills missing cells; never share moved mappings with selection.
    result.data_buffer[row] = (
      previous.data_buffer[source].copy()
      if top <= source < bottom
      else blank.data_buffer[row]
    )
    result.zero_width_escapes[row] = (
      previous.zero_width_escapes[source].copy()
      if top <= source < bottom
      else blank.zero_width_escapes[row]
    )
  return result


def _row_cost(before: _Row, after: _Row) -> int:
  """Estimate output from changed spans, without visiting painted cell mappings."""
  if before == after:
    return 0
  if len(before) != len(after) or any(
    len(old[1]) != len(new[1]) for old, new in zip(before, after)
  ):
    return max(
      sum(len(text) for _, text in before),
      sum(len(text) for _, text in after),
    )
  cost = 0
  for (old_style, old), (style, text) in zip(before, after):
    cost += (
      sum(map(operator.ne, old, text)) if style == old_style else len(text)
    )
  return cost


def _scroll_saves_output(
  previous: tuple[_Row, ...], rows: tuple[_Row, ...], shift: int
) -> bool:
  repaint = sum(_row_cost(old, new) for old, new in zip(previous, rows))
  exposed = rows[:shift] if shift > 0 else rows[shift:]
  # Shifted overlap was verified; only the newly exposed strip needs painting.
  scrolling = sum(_row_cost((), row) for row in exposed)
  # Window's positional last-line style also needs repair after moving rows.
  boundary = rows[-1] if shift > 0 else previous[-1]
  scrolling += _row_cost((), boundary)
  return scrolling < repaint


def install[T](app: application.Application[T]) -> None:
  """Select transcript rendering before the application's first paint."""
  previous = app.renderer
  if app.full_screen and type(previous) is renderer.Renderer:
    app.renderer = Renderer(
      previous.style,
      app.output,
      full_screen=True,
      mouse_support=previous.mouse_support,
      cpr_not_supported_callback=previous.cpr_not_supported_callback,
    )
