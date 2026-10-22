import dataclasses

import pytest
from rich import cells

from mirror.cli import actions, rendering, state, tool_groups
from mirror.cli.components import pending


@pytest.fixture
def snapshot() -> state.State:
  return state.State(
    activity=state.ActivityState(),
    status_bar=state.StatusBarState('test', 'medium', 'auto'),
    session=state.SessionState(conversation_id='test'),
    lifecycle=state.LifecycleState(active=True),
    layout=state.LayoutState(columns=40),
  )


def test_render_separates_consecutive_groups_and_retained_summary(
  snapshot: state.State,
) -> None:
  snapshot = dataclasses.replace(
    snapshot,
    tool_group=tool_groups.ToolGroup(
      key='tool:mcp', tool_counts=(('call', 1),)
    ),
    stream=state.StreamState(
      running=(
        actions.RunningToolCall('read', 'read', 'a.py', 'tool', '{}'),
        actions.RunningToolCall('grep', 'grep', 'needle', 'tool', '{}'),
        actions.RunningToolCall('bash', 'bash', 'echo ok', 'tool', '{}'),
      ),
      pending=(actions.PendingToolCall(0, 'read', 'b.py', 'tool'),),
    ),
  )
  rendered = rendering.render_blocks(
    pending.renderables(
      actions.ViewMode.DEFAULT,
      snapshot.stream.running,
      snapshot.stream.pending,
      snapshot.tool_group,
      'Ctrl+B',
    ),
    40,
  )
  assert [
    line.rstrip() for line in rendering.strip_ansi(rendered).splitlines()
  ] == [
    '·  mcp · 1 call',
    '',
    '·  Exploring',
    '   └ Searching needle · 2 running',
    '',
    '·  Running echo ok',
    '   └ Ctrl+B to background',
    '',
    '·  Exploring',
    '   └ Reading b.py',
  ]
  assert pending.rows(snapshot) == 9
  window = pending.Pending().render(snapshot)
  assert window.preferred_height(40, 24).preferred == 9


@pytest.mark.parametrize('columns', [24, 80])
@pytest.mark.parametrize('settled', [False, True])
def test_render_group_bounds_multiline_wide_arguments(
  snapshot: state.State, columns: int, settled: bool
) -> None:
  arguments = "界界\n\t'needle' " + '界' * 60
  snapshot = dataclasses.replace(
    snapshot,
    layout=state.LayoutState(columns=columns),
    tool_group=tool_groups.result('read', arguments, success=True)
    if settled
    else None,
    stream=state.StreamState(
      pending=()
      if settled
      else (actions.PendingToolCall(0, 'read', arguments, 'tool'),),
    ),
  )
  rendered = pending.render(snapshot)
  lines = rendering.strip_ansi(rendered.value).splitlines()
  assert len(lines) == pending.rows(snapshot) == 2
  assert max(map(cells.cell_len, lines)) <= columns
  assert lines[1].endswith('…')
  assert '界界 ' in lines[1]
  assert '\t' not in lines[1]
  bold = ''.join(
    value
    for style, value, *_ in rendered.__pt_formatted_text__()
    if 'bold' in style
  )
  assert 'Read' in bold


def test_render_detail_preserves_every_call(snapshot: state.State) -> None:
  snapshot = dataclasses.replace(
    snapshot,
    status_bar=dataclasses.replace(
      snapshot.status_bar, view_mode=actions.ViewMode.DETAIL
    ),
    stream=state.StreamState(
      running=(actions.RunningToolCall('read', 'read', 'a.py', 'tool', '{}'),),
      pending=(
        actions.PendingToolCall(0, 'bash', 'echo ok', 'tool'),
        actions.PendingToolCall(1, 'read', 'b.py', 'tool'),
      ),
    ),
  )
  rendered = rendering.strip_ansi(pending.render(snapshot).value)
  assert (
    rendered.index('read(a.py)')
    < rendered.index('bash(echo ok)')
    < rendered.index('read(b.py)')
  )
  assert rendered.count('running…') == 3
  assert pending.rows(snapshot) == 8
