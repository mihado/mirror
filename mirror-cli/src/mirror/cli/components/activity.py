"""Live activity and completed turn timing for the terminal UI."""

from __future__ import annotations

from typing import override

from prompt_toolkit import formatted_text
from prompt_toolkit.layout import containers, controls, dimension

from mirror.cli import rendering, state, tool_groups
from mirror.cli.components import component, spinner, text_shimmer


class Activity(component.Component):
  """The agent-activity row above the status bar."""

  @override
  def render(self, state: state.State) -> containers.AnyContainer:
    """Render the activity row from immutable application state."""
    fragments = _content(state)
    return containers.Window(
      controls.FormattedTextControl(fragments),
      height=dimension.Dimension.exact(1 if fragments else 0),
      style='class:activity',
    )


def _content(state: state.State) -> formatted_text.StyleAndTextTuples:
  pending_switch = state.session.pending_switch is not None
  busy = state.activity.busy
  status = state.activity.status
  if pending_switch or state.approval.request is not None:
    return []
  if not busy:
    duration = state.activity.completed_duration
    if duration is None:
      return []
    label = '<1s' if duration < 1 else rendering.format_duration(duration)
    return [
      (
        f'class:activity class:{state.activity.spinner_color.value}',
        rendering.icon_prefix(rendering.ASSISTANT_ICON),
      ),
      ('class:activity', f'Worked for {label}'),
    ]
  if not status:
    return []
  if state.turn.compaction_started is not None:
    status = 'Compacting'
  elif status in {'thinking…', 'running tool…'}:
    calls = (*state.stream.running, *state.stream.pending)
    key = state.tool_group.key if state.tool_group is not None else None
    if key is None and calls:
      keys = {tool_groups.group_key(call.name) for call in calls}
      if len(keys) == 1:
        key = keys.pop()
    status = tool_groups.status_label(key) if key is not None else 'Working'
  tokens = (
    f' ({_format_token_count(state.turn.output_tokens)} tokens)'
    if state.turn.output_tokens
    else ''
  )
  timer = ''
  if state.activity.work_started is not None:
    timer = f' · {rendering.Stopwatch(state.activity.work_started).render()}'
  indicator = spinner.render(
    state.activity.spinner_color, style='class:activity'
  )
  padding = rendering.GUTTER_WIDTH - formatted_text.fragment_list_width(
    indicator
  )
  return [
    *indicator,
    ('class:activity', ' ' * padding),
    *text_shimmer.render(status, style='class:activity'),
    ('class:activity', tokens + timer),
  ]


def _format_token_count(tokens: int) -> str:
  """Keep small counts exact; use lowercase k/m with one decimal above them."""
  if tokens > 1_000_000:
    return f'{tokens / 1_000_000:.1f}m'
  if tokens > 1_000:
    return f'{tokens / 1_000:.1f}k'
  return f'{tokens:,}'
