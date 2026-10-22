"""Bounded dock geometry and painting for prompt-toolkit rows and columns."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import override

from prompt_toolkit.application import current
from prompt_toolkit.layout import (
  containers,
  dimension,
  mouse_handlers,
  scrollable_pane,
)
from prompt_toolkit.layout import screen as screen_module

from mirror.cli import workspace_application


@functools.lru_cache(maxsize=1024)
def _allocate(
  dimensions: tuple[tuple[int, int, int], ...],
  available: int,
  extend: bool,
) -> tuple[int, ...] | None:
  """Allocate equal-weight rows/columns in whole rounds instead of cell by cell."""
  sizes: list[int] = [minimum for minimum, _, _ in dimensions]
  if sum(sizes) > available:
    return None
  cursor: int = 0
  for phase in (1, 2) if extend else (1,):
    limits: list[int] = [item[phase] for item in dimensions]
    remaining = min(available, sum(limits)) - sum(sizes)
    while remaining > 0:
      order: list[int] = [*range(cursor, len(sizes)), *range(cursor)]
      active: list[int] = [i for i in order if sizes[i] < limits[i]]
      rounds = min(
        remaining // len(active), min(limits[i] - sizes[i] for i in active)
      )
      if rounds:
        for i in active:
          sizes[i] += rounds
        remaining -= rounds * len(active)
        cursor = (active[-1] + 1) % len(sizes)
      else:
        for i in active[:remaining]:
          sizes[i] += 1
          remaining -= 1
          cursor = (i + 1) % len(sizes)
  return tuple(sizes)


class FrameCache[T]:
  """Share geometry during a draw, optionally retaining the previous draw's keys."""

  def __init__(self, *, retain: bool = False) -> None:
    self._frame: tuple[object, int] | None = None
    self._values: dict[tuple[int, ...], T] = {}
    self._used: set[tuple[int, ...]] = set()
    self._retain = retain

  def get(
    self,
    key: tuple[int, ...],
    calculate: Callable[[], T],
  ) -> T:
    app = current.get_app()
    if not self._retain and not (
      isinstance(app, workspace_application.WorkspaceApplication)
      and app.rendering
    ):
      self._frame = None
      self._values.clear()
      self._used.clear()
      return calculate()
    frame = (app, app.render_counter)
    if frame != self._frame:
      self._values = (
        {key: self._values[key] for key in self._used}
        if self._retain and self._frame is not None and self._frame[0] is app
        else {}
      )
      self._frame = frame
      self._used.clear()
    if key not in self._values:
      self._values[key] = calculate()
    if self._retain:
      self._used.add(key)
    return self._values[key]


class Columns(containers.VSplit):
  """Share column measurement within a frame without caching dynamic content."""

  def __init__(
    self, children: Sequence[containers.AnyContainer], *, retain: bool = False
  ) -> None:
    super().__init__(children)
    self._height_cache = FrameCache[dimension.Dimension](retain=retain)
    self._width_cache = FrameCache[list[int] | None](retain=retain)

  @override
  def preferred_height(
    self, width: int, max_available_height: int
  ) -> dimension.Dimension:
    measure = super().preferred_height
    return self._height_cache.get(
      (width, max_available_height),
      lambda: measure(width, max_available_height),
    )

  @override
  def _divide_widths(self, width: int) -> list[int] | None:
    return self._width_cache.get((width,), lambda: self._widths(width))

  def _widths(self, width: int) -> list[int] | None:
    sizes = [child.preferred_width(width) for child in self._all_children]
    if any(size.weight != 1 for size in sizes):
      return super()._divide_widths(width)
    allocated = _allocate(
      tuple((d.min, d.preferred, d.max) for d in sizes), width, True
    )
    return list(allocated) if allocated is not None else None


class Rows(containers.HSplit):
  """Share row measurement under the same frame boundary as columns."""

  def __init__(
    self,
    children: Sequence[containers.AnyContainer],
    *,
    height: dimension.AnyDimension = None,
    style: str = '',
    padding: dimension.AnyDimension = 0,
  ) -> None:
    super().__init__(children, height=height, style=style, padding=padding)
    self._height_cache = FrameCache[dimension.Dimension]()

  @override
  def preferred_height(
    self, width: int, max_available_height: int
  ) -> dimension.Dimension:
    measure = super().preferred_height
    return self._height_cache.get(
      () if self.height is not None else (width, max_available_height),
      lambda: measure(width, max_available_height),
    )

  @override
  def _divide_heights(
    self, write_position: screen_module.WritePosition
  ) -> list[int] | None:
    sizes = [
      child.preferred_height(write_position.width, write_position.height)
      for child in self._all_children
    ]
    if any(size.weight != 1 for size in sizes):
      return super()._divide_heights(write_position)
    allocated = _allocate(
      tuple((d.min, d.preferred, d.max) for d in sizes),
      write_position.height,
      not current.get_app().is_done,
    )
    return list(allocated) if allocated is not None else None


class ScrollableRows(scrollable_pane.ScrollablePane):
  """Paint fitting content directly; use the virtual canvas only for clipping."""

  @override
  def _copy_over_write_positions(
    self,
    screen: screen_module.Screen,
    temp_screen: screen_module.Screen,
    write_position: screen_module.WritePosition,
  ) -> None:
    # Stock ScrollablePane clips pixels but leaves window bounds unclipped.
    windows = temp_screen.visible_windows_to_write_positions
    for window, bounds in windows.items():
      left = max(write_position.xpos, write_position.xpos + bounds.xpos)
      top = max(
        write_position.ypos,
        write_position.ypos + bounds.ypos - self.vertical_scroll,
      )
      right = min(
        write_position.xpos + write_position.width,
        write_position.xpos + bounds.xpos + bounds.width,
      )
      bottom = min(
        write_position.ypos + write_position.height,
        write_position.ypos
        + bounds.ypos
        + bounds.height
        - self.vertical_scroll,
      )
      if left < right and top < bottom:
        screen.visible_windows_to_write_positions[window] = (
          screen_module.WritePosition(left, top, right - left, bottom - top)
        )

  @override
  def write_to_screen(
    self,
    screen: screen_module.Screen,
    mouse_handlers: mouse_handlers.MouseHandlers,
    write_position: screen_module.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    height = self.content.preferred_height(
      write_position.width, self.max_available_height
    ).preferred
    if (
      not self.show_scrollbar()
      and not self.vertical_scroll
      and height == write_position.height
    ):
      self.content.write_to_screen(
        screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
      )
    else:
      super().write_to_screen(
        screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
      )
