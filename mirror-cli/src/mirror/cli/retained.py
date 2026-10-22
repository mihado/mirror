"""Retained layout and painted chrome for transcript-only frame damage."""

from collections.abc import Callable
from typing import override

from prompt_toolkit.application import current
from prompt_toolkit.layout import containers, dimension
from prompt_toolkit.layout import mouse_handlers as mouse_handlers_module
from prompt_toolkit.layout import screen as screen_module

from mirror.cli import row_renderer, workspace_application


class Layer(containers.DynamicContainer):
  """Retain an opaque UI layer until its inputs or application revision change."""

  def __init__(
    self,
    content: containers.Container,
    geometry: Callable[[], object] = lambda: None,
    *,
    animated: bool = True,
  ) -> None:
    super().__init__(lambda: content)
    self._geometry = geometry
    self._animated = animated
    self._revision: tuple[object, object, object] | None = None
    self._heights: dict[tuple[int, int], dimension.Dimension] = {}
    self._widths: dict[int, dimension.Dimension] = {}
    self._position: tuple[int, int, int, int, str, bool, int | None] | None = (
      None
    )
    self._screen: screen_module.Screen | None = None
    self._handlers = row_renderer.MouseHandlers()

  def _prepare(self) -> None:
    app = current.get_app()
    revision = (
      (
        app,
        (app.ui_revision, app.animation_revision if self._animated else None),
        self._geometry(),
      )
      if isinstance(app, workspace_application.WorkspaceApplication)
      and app.is_running
      else None
    )
    if revision is None or revision != self._revision:
      self._revision = revision
      self._heights.clear()
      self._widths.clear()
      self._screen = None

  @override
  def reset(self) -> None:
    self._revision = None
    self._screen = None
    self._heights.clear()
    self._widths.clear()
    super().reset()

  @override
  def preferred_width(self, max_available_width: int) -> dimension.Dimension:
    self._prepare()
    if max_available_width not in self._widths:
      self._widths[max_available_width] = super().preferred_width(
        max_available_width
      )
    return self._widths[max_available_width]

  @override
  def preferred_height(
    self, width: int, max_available_height: int
  ) -> dimension.Dimension:
    self._prepare()
    key = width, max_available_height
    if key not in self._heights:
      self._heights[key] = super().preferred_height(width, max_available_height)
    return self._heights[key]

  @override
  def write_to_screen(
    self,
    screen: screen_module.Screen,
    mouse_handlers: mouse_handlers_module.MouseHandlers,
    write_position: screen_module.WritePosition,
    parent_style: str,
    erase_bg: bool,
    z_index: int | None,
  ) -> None:
    self._prepare()
    position = (
      write_position.xpos,
      write_position.ypos,
      write_position.width,
      write_position.height,
      parent_style,
      erase_bg,
      z_index,
    )
    if self._screen is None or position != self._position:
      painted = screen_module.Screen()
      painted.show_cursor = False
      self._handlers = row_renderer.MouseHandlers()
      super().write_to_screen(
        painted, self._handlers, write_position, parent_style, erase_bg, z_index
      )
      painted.draw_all_floats()
      self._screen, self._position = painted, position
    self._compose(screen, mouse_handlers)

  def _compose(
    self,
    target: screen_module.Screen,
    handlers: mouse_handlers_module.MouseHandlers,
  ) -> None:
    painted = self._screen
    if painted is None:
      return
    # Selection and overlays may mutate the composed frame, never this layer.
    for row, cells in painted.data_buffer.items():
      target.data_buffer[row].update(cells)
    for row, escapes in painted.zero_width_escapes.items():
      target.zero_width_escapes[row].update(escapes)
    if self._position is not None:
      self._handlers.copy_to(
        handlers,
        screen_module.WritePosition(*self._position[:4]),
        lambda handler: handler,
      )
    target.visible_windows_to_write_positions.update(
      painted.visible_windows_to_write_positions
    )
    target.cursor_positions.update(painted.cursor_positions)
    target.menu_positions.update(painted.menu_positions)
    target.show_cursor |= painted.show_cursor
    target.width = max(target.width, painted.width)
    target.height = max(target.height, painted.height)
