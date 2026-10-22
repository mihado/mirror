"""Declarative component bridge for prompt_toolkit."""

from __future__ import annotations

import abc
import types
from collections.abc import Callable, Mapping
from typing import override

from prompt_toolkit.application import current
from prompt_toolkit.layout import containers, dimension

from mirror.cli import state, workspace_application
from mirror.cli.components import layout


class Component(abc.ABC):
  """A stateless terminal component rendered from application state."""

  @abc.abstractmethod
  def render(self, state: state.State) -> containers.AnyContainer:
    """Render the component from one immutable state snapshot."""

  def cache_key(self, state: state.State) -> object | None:
    """Identify unchanged render inputs; None keeps clocks and callbacks live.

    Retained components must include every input used to construct their tree.
    Reuse lasts only while those inputs remain valid; arbitrary extension and
    animation output is frame-local.
    """
    del state
    return None


class ComponentCollection:
  """The fixed set of declarative components owned by one REPL."""

  def __init__(self, items: Mapping[str, Component]) -> None:
    self._items = types.MappingProxyType(dict(items))

  def __getitem__(self, name: str) -> Component:
    return self._items[name]


class ComponentHost:
  """Mount stateless components into prompt_toolkit's retained layout."""

  def __init__(self, get_state: Callable[[], state.State]) -> None:
    self._get_state = get_state

  def mount(self, component: Component) -> containers.AnyContainer:
    """Adapt one declarative component to a retained container."""
    return _Mounted(component, self._get_state)


class _Mounted(containers.DynamicContainer):
  """Own one component tree and its measurements while its inputs are unchanged."""

  def __init__(
    self, component: Component, get_state: Callable[[], state.State]
  ) -> None:
    self._component = component
    self._get_state = get_state
    self._snapshot: state.State | None = None
    self._app: object | None = None
    self._counter = -1
    self._key: object | None = None
    self._rendered: containers.Container | None = None
    self._heights = layout.FrameCache[dimension.Dimension]()
    self._widths = layout.FrameCache[dimension.Dimension]()
    super().__init__(self._container)

  def _container(self) -> containers.Container:
    app = current.get_app()
    counter = app.render_counter
    if (
      self._rendered is not None
      and app is self._app
      and counter == self._counter
      and isinstance(app, workspace_application.WorkspaceApplication)
      and app.rendering
    ):
      return self._rendered
    snapshot = self._get_state()
    if not app.is_running and counter == 0:
      return containers.to_container(self._component.render(snapshot))
    key = self._component.cache_key(snapshot)
    if (
      self._rendered is None
      or app is not self._app
      or key != self._key
      or key is None
      and (snapshot is not self._snapshot or counter != self._counter)
    ):
      self._rendered = containers.to_container(self._component.render(snapshot))
      self._heights = layout.FrameCache[dimension.Dimension](
        retain=key is not None
      )
      self._widths = layout.FrameCache[dimension.Dimension](
        retain=key is not None
      )
    self._snapshot, self._app, self._counter, self._key = (
      snapshot,
      app,
      counter,
      key,
    )
    return self._rendered

  @override
  def preferred_height(
    self, width: int, max_available_height: int
  ) -> dimension.Dimension:
    child = self._container()
    app = current.get_app()
    if not app.is_running and app.render_counter == 0:
      return child.preferred_height(width, max_available_height)
    return self._heights.get(
      (width, max_available_height),
      lambda: child.preferred_height(width, max_available_height),
    )

  @override
  def preferred_width(self, max_available_width: int) -> dimension.Dimension:
    child = self._container()
    app = current.get_app()
    if self._key is None or not app.is_running and app.render_counter == 0:
      return child.preferred_width(max_available_width)
    return self._widths.get(
      (max_available_width,), lambda: child.preferred_width(max_available_width)
    )
