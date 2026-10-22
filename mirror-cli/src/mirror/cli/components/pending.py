"""Stateless rows for streamed and running tool calls."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from collections.abc import Callable
from typing import override

from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.layout import containers, controls, dimension
from rich import console, table, text

from mirror.cli import actions as _actions
from mirror.cli import rendering, themes, tool_groups
from mirror.cli import state as _state
from mirror.cli.components import component


class Pending(component.Component):
  """Tool calls that are assembling or awaiting results."""

  def __init__(
    self,
    shortcut_hint: Callable[[_actions.Action | None], str] = lambda _: '',
  ) -> None:
    self._shortcut_hint = shortcut_hint

  @override
  def render(self, state: _state.State) -> containers.AnyContainer:
    shortcut = self._shortcut_hint(_actions.DetachRequested())
    return containers.Window(
      controls.FormattedTextControl(render(state, shortcut)),
      height=dimension.Dimension.exact(rows(state, shortcut)),
      wrap_lines=False,
    )


def render(state: _state.State, background_shortcut: str = '') -> ansi.ANSI:
  """Render in-flight tools with single-line summaries in the default view."""
  if state.approval.request is not None:
    return ansi.ANSI('')
  return _render(
    state.status_bar.view_mode,
    state.stream.running,
    state.stream.pending,
    state.tool_group,
    state.layout.columns,
    themes.current(state.extensions, state.extension_themes),
    background_shortcut if state.can_detach_tools else '',
  )


@functools.lru_cache(maxsize=16)
def _render(
  view_mode: _actions.ViewMode,
  running: tuple[_actions.RunningToolCall, ...],
  pending: tuple[_actions.PendingToolCall, ...],
  group: tool_groups.ToolGroup | None,
  columns: int,
  theme_name: str | themes.Theme,
  background_shortcut: str,
) -> ansi.ANSI:
  """Share tool-row rendering between measurement and painting."""
  blocks = renderables(view_mode, running, pending, group, background_shortcut)
  return ansi.ANSI(
    rendering.render_blocks(blocks, columns, theme_name=theme_name)
  )


def renderables(
  view_mode: _actions.ViewMode,
  running: tuple[_actions.RunningToolCall, ...],
  pending: tuple[_actions.PendingToolCall, ...],
  group: tool_groups.ToolGroup | None,
  background_shortcut: str = '',
) -> tuple[console.RenderableType, ...]:
  """Share live tool presentation between scrollback and fullscreen layouts."""
  if view_mode is _actions.ViewMode.FOCUS:
    return ()
  hint = f'{background_shortcut} to background' if background_shortcut else ''
  body = hint or 'running…'
  if view_mode is _actions.ViewMode.DEFAULT:
    parts: list[console.RenderableType] = []
    calls = (*running, *pending)
    # Only consecutive calls with the same semantic key share a progress row.
    batches = [
      tuple(batch)
      for _, batch in itertools.groupby(
        calls, key=lambda call: tool_groups.group_key(call.name)
      )
    ]
    if len(batches) > 1:
      hinted_call = next(
        (
          call
          for call in reversed(running)
          if tool_groups.group_key(call.name) != 'exploration'
        ),
        None,
      )
      for index, batch in enumerate(batches):
        parts.extend(
          renderables(
            view_mode,
            tuple(
              call
              for call in batch
              if isinstance(call, _actions.RunningToolCall)
            ),
            tuple(
              call
              for call in batch
              if isinstance(call, _actions.PendingToolCall)
            ),
            group if index == 0 else None,
            background_shortcut if hinted_call in batch else '',
          )
        )
      return tuple(parts)
    key = tool_groups.group_key(calls[-1].name) if calls else None
    if group is not None and calls and group.key != key:
      return (
        group.summary(),
        *renderables(view_mode, running, pending, None, background_shortcut),
      )
    summary = group
    if summary is None and key is not None:
      summary = tool_groups.ToolGroup(key=key)
    if summary is not None:
      parts.append(summary.summary(active=bool(calls)))
    label = group.latest if group is not None else None
    if calls:
      current = calls[-1]
      args = (
        rendering.command_preview(current.args)
        if current.name == 'bash'
        else current.args
      )
      verb = {
        'bash': 'Running',
        'read': 'Reading',
        'grep': 'Searching',
        'find': 'Searching',
        'glob': 'Searching',
        'ls': 'Scanning',
        'edit': 'Editing',
        'write': 'Writing',
        'web_search': 'Web searching',
        'web_browse': 'Web browsing',
        'subagent': 'Subagent',
      }.get(current.name, tool_groups.display_name(current.name).capitalize())
      label = rendering.ToolLabel(
        ' '.join(verb.split()),
        ' '.join(args.split()),
        f' · {len(calls)} running' if len(calls) > 1 else '',
      )
    if label is not None:
      if summary is not None:
        label = dataclasses.replace(
          label,
          arguments=' '.join(rendering.strip_ansi(label.arguments).split()),
        )
      title = label.render()
      title.overflow = 'ellipsis'
      title.no_wrap = True
      if summary is None:
        return (
          rendering.tool_block(
            title,
            rendering.result_summary(hint) if hint and running else None,
          ),
        )
      prefix = ' ' * rendering.GUTTER_WIDTH + '└ '
      row = table.Table.grid()
      row.add_column(width=len(prefix))
      row.add_column()
      row.add_row(
        text.Text(prefix, style='dim'),
        title,
      )
      if hint and running and summary.key != 'exploration':
        row.add_row(
          text.Text(prefix, style='dim'), rendering.result_summary(hint)
        )
      parts.append(row)
    return (console.Group(*parts),) if parts else ()
  return (
    *(
      rendering.running_tool_renderable(*call.render_hint, body)
      for call in running
    ),
    *(
      rendering.running_tool_renderable(*call.render_hint, 'running…')
      for call in pending
    ),
  )


def rows(state: _state.State, background_shortcut: str = '') -> int:
  """Let the approval preview own tool presentation while a decision is pending."""
  if (
    state.status_bar.view_mode is _actions.ViewMode.FOCUS
    or state.approval.request is not None
  ):
    return 0
  has_running = bool(state.stream.running)
  has_summary = (
    state.status_bar.view_mode is _actions.ViewMode.DEFAULT
    and state.tool_group is not None
  )
  if not has_running and not state.stream.pending and not has_summary:
    return 0
  text = ''.join(
    fragment[1]
    for fragment in render(state, background_shortcut).__pt_formatted_text__()
    if fragment[0] != '[ZeroWidthEscape]'
  )
  return text.count('\n') + 1


def count(state: _state.State) -> int:
  """Return the number of durable and still-streaming tool calls."""
  if state.status_bar.view_mode is _actions.ViewMode.FOCUS:
    return 0
  return len(state.stream.running) + len(state.stream.pending)
