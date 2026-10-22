"""Product events observed by one CLI session, without conversation content.

The worker stream already supplies lifecycle events and durable record identity;
this observer keeps analytics out of rendering and the pure state reducer.
"""

from __future__ import annotations

import dataclasses
import time
import uuid

from mirror.protocol import agent_pb2

from mirror.cli.statsig.events import client, event


@dataclasses.dataclass
class _Turn:
  id: str
  collection_epoch: int
  interrupt_requested: bool = False
  steering_submitted: bool = False
  observation_complete: bool = True
  failed: bool = False


@dataclasses.dataclass
class _ToolCall:
  name: str
  success: bool | None = None


class ConversationMetrics:
  """Count live activity once per attached conversation, excluding replay."""

  def __init__(
    self, *, remote: bool, events: client.EventClient | None = None
  ) -> None:
    self._client = events if events is not None else client.get()
    self._turn: _Turn | None = None
    self._workspace_kind = 'remote' if remote else 'local'
    self._conversation_id = ''
    self._turn_active = False
    self._seen_entries: set[str] = set()
    self._seen_turns: set[str] = set()
    self._compaction_started: float | None = None
    self._tool_calls: dict[str, _ToolCall] = {}

  def observe(
    self,
    message: agent_pb2.SessionMessage,
    *,
    conversation_id: str,
    replaying: bool,
    turn_duration_seconds: float | None = None,
  ) -> None:
    """Observe authoritative client events before their UI state changes."""
    if conversation_id != self._conversation_id:
      self._seen_entries.clear()
      self._seen_turns.clear()
      self._turn = None
    if conversation_id != self._conversation_id or replaying:
      self.suspend()
      self._compaction_started = None
      self._tool_calls.clear()
    self._conversation_id = conversation_id
    if message.HasField('agent_message'):
      entry = message.agent_message
      if replaying and entry.turn_id:
        self._seen_turns.add(entry.turn_id)
      if not self._new_entry(entry) or replaying:
        return
      self._entry(entry)
    if replaying:
      return
    if message.WhichOneof('payload') != 'runtime_delta':
      return
    kind = message.runtime_delta.WhichOneof('payload')
    if kind == 'turn_error':
      self._compaction_started = None
      turn = self._eligible_turn()
      if turn is not None:
        turn.failed = True
    if kind == 'compaction_started' and self._compaction_started is None:
      self._compaction_started = time.monotonic()
    if kind == 'turn_started':
      self._start(message.runtime_delta)
    elif kind in {'turn_completed', 'turn_interrupted'}:
      if self._turn_active:
        if turn_duration_seconds is not None:
          self._emit('mirror_turn_duration_seconds', turn_duration_seconds)
        outcome = (
          'worker_interrupted'
          if kind == 'turn_interrupted'
          else 'failed'
          if self._turn is not None and self._turn.failed
          else 'completed'
        )
        self._emit(
          'mirror_turn_tool_call_count',
          len(self._tool_calls),
          outcome=outcome,
        )
        self._emit(
          'mirror_turn_tool_success_count',
          sum(call.success is True for call in self._tool_calls.values()),
        )
        self._emit(
          'mirror_turn_tool_failure_count',
          sum(call.success is False for call in self._tool_calls.values()),
        )
        self._tool_calls.clear()
      if kind == 'turn_interrupted' and self._turn_active:
        self._emit('mirror_agent_turn_interrupted')
      self._turn_active = False
      self._compaction_started = None

  def suspend(self) -> None:
    """Keep identity across recovery, but never claim gap-free observation."""
    self._turn_active = False
    if self._turn is not None:
      self._turn.observation_complete = False

  def interrupt_requested(self) -> None:
    """Observe Esc routed to interruption, not dismissal or another shortcut."""
    turn = self._eligible_turn()
    if turn is not None and not turn.interrupt_requested:
      turn.interrupt_requested = True
      self._emit('mirror_turn_interrupt_requested')

  def tools_background_requested(self) -> None:
    """Count actionable background requests, not automatic task detachment."""
    self._emit('mirror_tools_background_requested')

  def steering_submitted(self) -> None:
    """Observe human mid-turn submission, independent of worker acceptance."""
    turn = self._eligible_turn()
    if turn is not None and not turn.steering_submitted:
      turn.steering_submitted = True
      self._emit('mirror_message_queued')

  def _eligible_turn(self) -> _Turn | None:
    turn = self._turn
    if (
      self._turn_active
      and turn is not None
      and self._client.enabled
      and turn.collection_epoch == self._client.collection_epoch
    ):
      return turn
    return None

  def _start(self, delta: agent_pb2.RuntimeDelta) -> None:
    try:
      turn_id = uuid.UUID(delta.turn_id).hex
    except ValueError:
      turn_id = ''
    if turn_id and self._turn is not None and self._turn.id == turn_id:
      if not self._turn_active:
        self._turn.observation_complete = False
        self._turn.failed = False
      self._turn_active = True
      return
    if not turn_id and self._turn_active:
      return
    self._turn = None
    self._turn_active = True
    self._tool_calls.clear()
    already_seen = bool(turn_id) and delta.turn_id in self._seen_turns
    if turn_id:
      self._seen_turns.add(delta.turn_id)
      self._turn = _Turn(
        turn_id,
        self._client.collection_epoch if self._client.enabled else -1,
        observation_complete=not already_seen,
      )
    if not already_seen:
      self._emit('mirror_conversation_turn_started')

  def _new_entry(self, entry: agent_pb2.AgentMessage) -> bool:
    if entry.id:
      if entry.id in self._seen_entries:
        return False
      self._seen_entries.add(entry.id)
    return True

  def _entry(self, entry: agent_pb2.AgentMessage) -> None:
    if entry.HasField('context_window') and entry.context_window.HasField(
      'compaction'
    ):
      self._emit('mirror_compaction_completed')
      if self._compaction_started is not None:
        self._emit(
          'mirror_compaction_duration_seconds',
          time.monotonic() - self._compaction_started,
        )
      self._compaction_started = None
    if entry.WhichOneof('payload') != 'chat':
      return
    chat = entry.chat
    if chat.WhichOneof('message') == 'user' and not chat.HasField('display'):
      self._emit('mirror_conversation_user_message')
    elif chat.WhichOneof('message') == 'assistant':
      self._emit('mirror_conversation_assistant_message')
      if self._turn_active:
        for call in chat.assistant.tool_calls:
          self._tool_calls.setdefault(call.id, _ToolCall(call.function.name))
    elif chat.WhichOneof('message') == 'tool':
      tracked_call = self._tool_calls.get(chat.tool.tool_call_id)
      if tracked_call is not None:
        tracked_call.success = chat.success
        if self._eligible_turn() is not None and entry.metadata.HasField(
          'tool_duration_seconds'
        ):
          duration = entry.metadata.tool_duration_seconds
          if 0 <= duration < float('inf'):
            self._emit(
              'mirror_tool_call_duration_seconds',
              duration,
              tool_name=tracked_call.name,
            )

  def _emit(
    self,
    name: str,
    value: int | float = 1,
    *,
    outcome: str | None = None,
    tool_name: str | None = None,
  ) -> None:
    metadata: dict[str, event.MetadataValue] = {
      'conversation_id': self._conversation_id,
      'workspace_kind': self._workspace_kind,
    }
    if tool_name:
      metadata['tool_name'] = tool_name
    turn = self._eligible_turn()
    if turn is not None and name in {
      'mirror_conversation_turn_started',
      'mirror_turn_interrupt_requested',
      'mirror_message_queued',
      'mirror_turn_tool_call_count',
    }:
      metadata.update(
        turn_metrics_version=1,
        turn_id=turn.id,
        interrupt_requested=turn.interrupt_requested,
        steering_submitted=turn.steering_submitted,
        observation_complete=turn.observation_complete,
      )
      if outcome is not None:
        metadata['outcome'] = outcome
    self._client.log_event(event.Event(name, value=value, metadata=metadata))
