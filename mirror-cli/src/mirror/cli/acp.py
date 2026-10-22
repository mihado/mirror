"""Serve the Agent Client Protocol on stdio over local agent workers.

ACP (https://agentclientprotocol.com) is JSON-RPC 2.0 as newline-delimited
JSON on stdin/stdout. Clients such as Omnigent and Zed use it to drive any
coding agent. `mirror acp` maps one ACP session onto one local agent worker:
`session/new` starts a worker in the client's working directory,
`session/prompt` runs one turn and streams its deltas back, tool approvals
become `session/request_permission` requests, and `session/cancel` interrupts
the turn. Stdout carries the protocol; diagnostics go to stderr.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import json
import os
import pathlib
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from mirror.protocol import agent_pb2

from mirror.cli import cli_store, local_rpc_client, noninteractive, onboarding
from mirror.cli import rpc_client

PROTOCOL_VERSION = 1

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603

# One inbound line; prompts embed file context, so the bound is generous.
_MAX_LINE_BYTES = 64 * 1024 * 1024
# Tool output forwarded in a tool card; a full file read would otherwise be
# copied into every client transcript.
_MAX_TOOL_OUTPUT_CHARS = 16 * 1024
_PERMISSION_OPTIONS = (
  {'optionId': 'allow', 'name': 'Allow', 'kind': 'allow_once'},
  {'optionId': 'reject', 'name': 'Reject', 'kind': 'reject_once'},
)

type JSON = dict[str, object]
type WorkerConfigFactory = Callable[[str], Mapping[str, object]]
type TransportFactory = Callable[
  [Mapping[str, object], pathlib.Path, str, rpc_client.MessageHandler],
  rpc_client.Transport,
]


class RPCError(Exception):
  """A JSON-RPC error answered to the client."""

  def __init__(self, code: int, message: str) -> None:
    super().__init__(message)
    self.code = code
    self.message = message


class Output(Protocol):
  """Where protocol lines go; `sys.stdout.buffer` in production."""

  def write(self, data: bytes, /) -> object: ...
  def flush(self) -> object: ...


def _local_transport(
  worker_command: Sequence[str],
) -> TransportFactory:
  def factory(
    config: Mapping[str, object],
    workspace: pathlib.Path,
    conversation_id: str,
    on_message: rpc_client.MessageHandler,
  ) -> rpc_client.Transport:
    return local_rpc_client.Transport(
      worker_command,
      config,
      workspace=workspace,
      log_dir=pathlib.Path.home() / '.mirror' / 'logs' / conversation_id,
      on_message=on_message,
    )

  return factory


def tool_kind(name: str) -> str:
  """Map a Mirror tool name onto ACP's tool-call kinds (icons only)."""
  lowered = name.strip().lower()
  if lowered == 'read':
    return 'read'
  if lowered in ('edit', 'write', 'multi_edit'):
    return 'edit'
  if lowered in ('bash', 'shell', 'exec'):
    return 'execute'
  if lowered in ('grep', 'find', 'glob', 'ls'):
    return 'search'
  if lowered in ('web_search', 'web_browse', 'fetch', 'web_fetch'):
    return 'fetch'
  return 'other'


def prompt_text(blocks: Sequence[object]) -> str:
  """Flatten ACP prompt blocks into the one message a turn accepts."""
  parts: list[str] = []
  for value in blocks:
    block = _object(value)
    if block is None:
      continue
    kind = block.get('type')
    if kind == 'text':
      text = block.get('text')
      if isinstance(text, str) and text:
        parts.append(text)
    elif kind == 'resource':
      resource = _object(block.get('resource'))
      if resource is not None:
        text = resource.get('text')
        uri = resource.get('uri', '')
        if isinstance(text, str) and text:
          parts.append(f'<file uri="{uri}">\n{text}\n</file>')
    elif kind == 'resource_link':
      uri = str(block.get('uri', ''))
      name = block.get('name')
      reference = f'{name} ({uri})' if isinstance(name, str) and name else uri
      if reference:
        parts.append(f'Referenced resource: {reference}')
    elif kind in ('image', 'audio'):
      parts.append(f'[{kind} attachment omitted: not supported by mirror acp]')
  return '\n\n'.join(parts)


def _object(value: object) -> JSON | None:
  """Narrow a decoded JSON value to an object."""
  return cast(JSON, value) if isinstance(value, dict) else None


def _array(value: object) -> list[object]:
  """Narrow a decoded JSON value to an array; anything else is empty."""
  return cast(list[object], value) if isinstance(value, list) else []


def _tool_arguments(arguments: str) -> JSON:
  try:
    parsed: object = json.loads(arguments) if arguments.strip() else {}
  except ValueError:
    return {'arguments': arguments}
  parsed_object = _object(parsed)
  # An empty object is a valid, empty argument list.
  return parsed_object if parsed_object is not None else {'arguments': parsed}


def _truncate(text: str) -> str:
  if len(text) <= _MAX_TOOL_OUTPUT_CHARS:
    return text
  return text[:_MAX_TOOL_OUTPUT_CHARS] + '\n… [truncated]'


def _tool_call_update(
  call_id: str, name: str, arguments: str, status: str
) -> JSON:
  return {
    'sessionUpdate': 'tool_call',
    'toolCallId': call_id,
    'title': name,
    'kind': tool_kind(name),
    'status': status,
    'rawInput': _tool_arguments(arguments),
  }


def _tool_result_update(call_id: str, content: str, *, success: bool) -> JSON:
  update: JSON = {
    'sessionUpdate': 'tool_call_update',
    'toolCallId': call_id,
    'status': 'completed' if success else 'failed',
  }
  text = _truncate(content)
  if text:
    update['content'] = [
      {'type': 'content', 'content': {'type': 'text', 'text': text}}
    ]
  return update


def _chunk(kind: str, text: str) -> JSON:
  return {'sessionUpdate': kind, 'content': {'type': 'text', 'text': text}}


@dataclasses.dataclass
class TurnOutcome:
  """How one prompt ended."""

  stop_reason: str
  usage: JSON | None = None
  error: str | None = None


class Translator:
  """Translate one turn's session messages into ACP updates.

  Streamed deltas are forwarded as they arrive; committed records supply the
  tool calls and results, whose ids tie `tool_call` and `tool_call_update`
  together. Usage from the turn's model calls is summed for the final result.
  """

  def __init__(self) -> None:
    self._usage: dict[str, int] = {}
    self._task_titles: dict[str, str] = {}
    self._streamed = False
    self._interrupted = False
    self._notice: str | None = None

  def translate(
    self, event: agent_pb2.SessionMessage
  ) -> tuple[list[JSON], TurnOutcome | None]:
    payload = event.WhichOneof('payload')
    if payload == 'runtime_delta':
      return self._runtime_delta(event.runtime_delta)
    if payload == 'agent_message':
      return self._record(event.agent_message), None
    if payload == 'input_nack':
      return [], TurnOutcome(
        'end_turn',
        error=f'agent rejected the request: {event.input_nack.error}',
      )
    return [], None

  def usage(self) -> JSON | None:
    if not self._usage:
      return None
    result: JSON = {
      'inputTokens': self._usage.get('prompt_tokens', 0),
      'outputTokens': self._usage.get('completion_tokens', 0),
      'totalTokens': self._usage.get('prompt_tokens', 0)
      + self._usage.get('completion_tokens', 0),
    }
    if cached := self._usage.get('cached_tokens'):
      result['cachedReadTokens'] = cached
    if written := self._usage.get('cache_write_tokens'):
      result['cachedWriteTokens'] = written
    return result

  def _runtime_delta(
    self, delta: agent_pb2.RuntimeDelta
  ) -> tuple[list[JSON], TurnOutcome | None]:
    kind = delta.WhichOneof('payload')
    if kind == 'stream_content':
      text = delta.stream_content.content
      if not text:
        return [], None
      self._streamed = True
      return [_chunk('agent_message_chunk', text)], None
    if kind == 'stream_reasoning':
      text = delta.stream_reasoning.reasoning
      return ([_chunk('agent_thought_chunk', text)] if text else []), None
    if kind == 'stream_refusal':
      text = delta.stream_refusal.refusal
      if not text:
        return [], None
      self._streamed = True
      return [_chunk('agent_message_chunk', text)], None
    if kind == 'context_usage':
      usage = delta.context_usage
      return [
        {
          'sessionUpdate': 'usage_update',
          'used': usage.tokens,
          'size': usage.window,
        }
      ], None
    if kind == 'task_update':
      return self._task_update(delta.task_update), None
    if kind == 'turn_completed':
      if self._interrupted:
        return [], TurnOutcome('cancelled', usage=self.usage())
      if self._notice is not None:
        # The worker gave up (for example on provider errors) and committed
        # a notice instead of a model answer.
        return [], TurnOutcome('end_turn', error=self._notice)
      return [], TurnOutcome('end_turn', usage=self.usage())
    if kind == 'turn_interrupted':
      # The worker follows this with `turn_completed`; ending the turn there
      # keeps that completion out of the next prompt's queue.
      self._interrupted = True
      return [], None
    if kind == 'turn_error':
      # Provider errors may contain request bodies or credentials.
      return [], TurnOutcome(
        'end_turn', error='agent turn failed; see the worker log for details'
      )
    return [], None

  def _task_update(self, task: agent_pb2.TaskUpdate) -> list[JSON]:
    call_id = f'task:{task.id}'
    title = task.display or task.tool_name or 'task'
    if task.status == agent_pb2.TASK_STATUS_RUNNING:
      if call_id in self._task_titles:
        return []
      self._task_titles[call_id] = title
      return [
        {
          'sessionUpdate': 'tool_call',
          'toolCallId': call_id,
          'title': title,
          'kind': tool_kind(task.tool_name),
          'status': 'in_progress',
        }
      ]
    if task.status in (
      agent_pb2.TASK_STATUS_FAILED,
      agent_pb2.TASK_STATUS_CANCELLED,
    ):
      self._task_titles.pop(call_id, None)
      return [
        {
          'sessionUpdate': 'tool_call_update',
          'toolCallId': call_id,
          'status': 'failed',
        }
      ]
    # Success carries its content in the task_result record that follows.
    return []

  def _record(self, record: agent_pb2.AgentMessage) -> list[JSON]:
    payload = record.WhichOneof('payload')
    if payload == 'task_result':
      result = record.task_result
      self._task_titles.pop(f'task:{result.task_id}', None)
      return [
        _tool_result_update(
          f'task:{result.task_id}', result.content, success=result.success
        )
      ]
    if payload != 'chat':
      return []
    chat = record.chat
    if chat.HasField('usage'):
      for field in (
        'prompt_tokens',
        'completion_tokens',
        'cached_tokens',
        'cache_write_tokens',
      ):
        self._usage[field] = self._usage.get(field, 0) + getattr(
          chat.usage, field
        )
    message = chat.WhichOneof('message')
    if message == 'assistant':
      return self._assistant(record)
    if message == 'tool':
      tool = chat.tool
      if not tool.tool_call_id:
        return []
      return [
        _tool_result_update(
          tool.tool_call_id, tool.content, success=chat.success
        )
      ]
    return []

  def _assistant(self, record: agent_pb2.AgentMessage) -> list[JSON]:
    chat = record.chat
    assistant = chat.assistant
    text = assistant.content or assistant.refusal
    if not record.HasField('inference'):
      # A synthetic notice (limits, provider failures), not a model answer.
      if text.strip() and not assistant.tool_calls:
        self._notice = text.strip()
      return []
    if chat.success:
      # A model answer after a notice means the worker recovered from it;
      # only a notice that ends the turn fails the prompt.
      self._notice = None
    updates: list[JSON] = []
    if text and not self._streamed:
      # Streaming was off or the provider sent no deltas: the committed
      # answer is the only copy of the text.
      updates.append(_chunk('agent_message_chunk', text))
    self._streamed = False
    updates.extend(
      _tool_call_update(
        call.id,
        call.function.name,
        call.function.arguments,
        'in_progress',
      )
      for call in assistant.tool_calls
      if call.id
    )
    return updates


@dataclasses.dataclass
class _Turn:
  """One accepted prompt, from dispatch until its response is written."""

  started: bool = False
  cancel_requested: bool = False
  # The outstanding `session/request_permission`, so a cancel can settle it.
  permission: asyncio.Future[JSON] | None = None


class _Session:
  """One ACP session: a worker, its event queue, and the active turn."""

  def __init__(
    self,
    session_id: str,
    client: rpc_client.Client,
    workspace: pathlib.Path,
    model: str,
  ) -> None:
    self.session_id = session_id
    self.client = client
    self.workspace = workspace
    self.model = model
    self.events: asyncio.Queue[agent_pb2.SessionMessage | None] = (
      asyncio.Queue()
    )
    self.turn: _Turn | None = None
    # Set once the worker connection is lost; the session takes no more prompts.
    self.failed = False


class Server:
  """Answer one ACP client over a line-oriented reader and writer."""

  def __init__(
    self,
    *,
    worker_config: WorkerConfigFactory,
    transport: TransportFactory,
    reasoning_effort: str,
    consent: Callable[[pathlib.Path], None] | None = None,
    record_history: bool = True,
  ) -> None:
    self._worker_config = worker_config
    self._transport = transport
    self._reasoning_effort = reasoning_effort
    self._consent = consent
    self._record_history = record_history
    self._sessions: dict[str, _Session] = {}
    self._pending: dict[str, asyncio.Future[JSON]] = {}
    self._next_request_id = 0
    self._output: Output | None = None
    self._write_lock = asyncio.Lock()
    self._requests: set[asyncio.Task[None]] = set()

  async def serve(self, reader: asyncio.StreamReader, output: Output) -> None:
    """Dispatch client messages until the client closes its end."""
    self._output = output
    try:
      while True:
        try:
          line = await reader.readline()
        except (asyncio.LimitOverrunError, ValueError):
          await self._write_error(None, _PARSE_ERROR, 'line too long')
          continue
        if not line:
          return
        line = line.strip()
        if not line:
          continue
        try:
          message = json.loads(line)
        except ValueError as error:
          await self._write_error(None, _PARSE_ERROR, f'invalid json: {error}')
          continue
        if not isinstance(message, dict):
          await self._write_error(None, _INVALID_REQUEST, 'expected an object')
          continue
        await self._dispatch(cast(JSON, message))
    finally:
      for task in list(self._requests):
        task.cancel()
      await asyncio.gather(*self._requests, return_exceptions=True)
      for session in list(self._sessions.values()):
        await session.client.close()
      self._sessions.clear()

  async def _dispatch(self, message: JSON) -> None:
    method = message.get('method')
    if not isinstance(method, str):
      request_id = message.get('id')
      if isinstance(request_id, str) and request_id in self._pending:
        # The client answered one of our requests (a permission prompt). A
        # cancel may have settled it already, with the answer close behind.
        future = self._pending.pop(request_id)
        if not future.done():
          future.set_result(message)
      return
    params = message.get('params')
    params = cast(JSON, params) if isinstance(params, dict) else {}
    request_id = message.get('id')
    if request_id is None:
      await self._notification(method, params)
      return
    if method == 'session/prompt':
      try:
        session, turn, text = self._accept_prompt(params)
      except RPCError as error:
        await self._write_error(request_id, error.code, error.message)
        return
      handler = self._prompt_request(request_id, session, turn, text)
    else:
      handler = self._respond(request_id, method, params)
    # Requests run off the read loop, so a slow worker start or a long turn
    # never delays a cancel or a permission answer for another session.
    task = asyncio.create_task(handler)
    self._requests.add(task)
    task.add_done_callback(self._requests.discard)

  async def _respond(
    self, request_id: object, method: str, params: JSON
  ) -> None:
    try:
      result = await self._request(method, params)
    except RPCError as error:
      await self._write_error(request_id, error.code, error.message)
      return
    await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

  async def _request(self, method: str, params: JSON) -> JSON:
    if method == 'initialize':
      return {
        'protocolVersion': PROTOCOL_VERSION,
        'agentCapabilities': {
          'loadSession': False,
          'promptCapabilities': {
            'image': False,
            'audio': False,
            'embeddedContext': True,
          },
        },
        'authMethods': [],
      }
    if method == 'authenticate':
      return {}
    if method == 'session/new':
      return await self._new_session(params)
    raise RPCError(_METHOD_NOT_FOUND, f'method not supported: {method}')

  async def _notification(self, method: str, params: JSON) -> None:
    if method != 'session/cancel':
      return
    session = self._sessions.get(str(params.get('sessionId', '')))
    if session is None or session.turn is None:
      return
    turn = session.turn
    turn.cancel_requested = True
    if turn.permission is not None and not turn.permission.done():
      # The client may never answer a permission request it has dismissed.
      turn.permission.set_result({
        'result': {'outcome': {'outcome': 'cancelled'}}
      })
    if turn.started:
      session.client.send(
        agent_pb2.ClientMessage(
          session_id=session.session_id,
          message_id=uuid.uuid4().hex,
          interrupt=agent_pb2.Interrupt(),
        )
      )

  async def _new_session(self, params: JSON) -> JSON:
    cwd = params.get('cwd')
    workspace = (
      pathlib.Path(cwd) if isinstance(cwd, str) and cwd else pathlib.Path.cwd()
    )
    if not workspace.is_dir():
      raise RPCError(_INVALID_PARAMS, f'cwd is not a directory: {workspace}')
    workspace = workspace.resolve()
    if self._consent is not None:
      try:
        self._consent(workspace)
      except SystemExit as error:
        raise RPCError(_INVALID_PARAMS, str(error)) from None
    servers = _array(params.get('mcpServers'))
    if servers:
      print(
        f'mirror acp: ignoring {len(servers)} mcpServers; configure MCP in '
        'Mirror instead',
        file=sys.stderr,
      )
    conversation_id = uuid.uuid4().hex
    config = self._worker_config(conversation_id)
    configuration = rpc_client.WorkerConfiguration(config, cwd=workspace)
    model = str(configuration.values['model'])
    events: asyncio.Queue[agent_pb2.SessionMessage | None] = asyncio.Queue()

    def on_message(event: agent_pb2.SessionMessage) -> None:
      events.put_nowait(event)
      if self._record_history:
        _record_user_message(event, conversation_id, workspace, model)

    async def on_error(error: str) -> None:
      del error  # Raw transport errors can include provider payloads.
      events.put_nowait(None)
      failed = self._sessions.get(conversation_id)
      if failed is None:
        return
      failed.failed = True
      turn = failed.turn
      if turn is not None and turn.permission is not None:
        # Nothing can approve a call on a worker that is gone.
        if not turn.permission.done():
          turn.permission.set_result({})

    transport = self._transport(
      configuration.values, workspace, conversation_id, on_message
    )
    client = rpc_client.Client(
      transport,
      conversation_id,
      on_message=on_message,
      on_error=on_error,
      web_provider=configuration.web_provider,
      inference_config=configuration.inference_config(
        model, self._reasoning_effort, remote=False
      ),
    )
    try:
      await asyncio.wait_for(client.start(replay=False), timeout=30)
    except (OSError, RuntimeError, TimeoutError) as error:
      await client.close()
      raise RPCError(
        _INTERNAL_ERROR,
        'could not start agent; check model authentication and worker '
        f'configuration ({error})',
      ) from None
    session = _Session(conversation_id, client, workspace, model)
    session.events = events
    self._sessions[conversation_id] = session
    print(
      f'mirror acp: session {conversation_id} in {workspace} ({model})',
      file=sys.stderr,
    )
    return {
      'sessionId': conversation_id,
      'models': {
        'availableModels': [{'modelId': model, 'name': model}],
        'currentModelId': model,
      },
    }

  def _accept_prompt(self, params: JSON) -> tuple[_Session, _Turn, str]:
    """Register the turn before its task runs, so a cancel cannot miss it."""
    session = self._sessions.get(str(params.get('sessionId', '')))
    if session is None:
      raise RPCError(_INVALID_PARAMS, 'unknown session')
    if session.failed:
      raise RPCError(
        _INTERNAL_ERROR, 'agent connection failed; start a new session'
      )
    text = prompt_text(_array(params.get('prompt')))
    if not text.strip():
      raise RPCError(_INVALID_PARAMS, 'prompt has no text content')
    if session.turn is not None:
      raise RPCError(_INVALID_REQUEST, 'a prompt is already running')
    session.turn = _Turn()
    return session, session.turn, text

  async def _prompt_request(
    self, request_id: object, session: _Session, turn: _Turn, text: str
  ) -> None:
    try:
      result = await self._run_turn(session, turn, text)
    except RPCError as error:
      await self._write_error(request_id, error.code, error.message)
      return
    except asyncio.CancelledError:
      await self._write_error(request_id, _INTERNAL_ERROR, 'server shut down')
      raise
    finally:
      if session.turn is turn:
        session.turn = None
    await self._write({'jsonrpc': '2.0', 'id': request_id, 'result': result})

  async def _run_turn(self, session: _Session, turn: _Turn, text: str) -> JSON:
    if turn.cancel_requested:
      # Cancelled before the worker saw the prompt.
      return {'stopReason': 'cancelled'}
    session.client.enqueue(
      agent_pb2.AgentMessage(
        id=uuid.uuid4().hex,
        chat=agent_pb2.ChatRecord(
          user=agent_pb2.UserMessage(content=text), success=True
        ),
      )
    )
    turn.started = True
    translator = Translator()
    while True:
      event = await session.events.get()
      if event is None:
        raise RPCError(
          _INTERNAL_ERROR,
          'agent connection failed; check provider configuration',
        )
      if event.WhichOneof('payload') == 'hitl_request':
        # A request already queued when the cancel or the disconnect landed:
        # nothing can act on the answer, so do not ask the client.
        if not turn.cancel_requested and not session.failed:
          await self._permission(session, turn, event.hitl_request.call)
        continue
      if _DEBUG:
        _debug_event(event)
      updates, outcome = translator.translate(event)
      for update in updates:
        await self._notify(
          'session/update', {'sessionId': session.session_id, 'update': update}
        )
      if outcome is None:
        continue
      if outcome.error is not None:
        raise RPCError(_INTERNAL_ERROR, outcome.error)
      result: JSON = {'stopReason': outcome.stop_reason}
      if outcome.usage is not None:
        result['usage'] = outcome.usage
      return result

  async def _permission(
    self, session: _Session, turn: _Turn, call: agent_pb2.ToolCall
  ) -> None:
    """Ask the client to approve a tool call and forward its answer."""
    self._next_request_id += 1
    request_id = f'mirror-{self._next_request_id}'
    future: asyncio.Future[JSON] = asyncio.get_running_loop().create_future()
    self._pending[request_id] = future
    turn.permission = future
    try:
      await self._write({
        'jsonrpc': '2.0',
        'id': request_id,
        'method': 'session/request_permission',
        'params': {
          'sessionId': session.session_id,
          'toolCall': _tool_call_update(
            call.id,
            call.function.name,
            call.function.arguments,
            'pending',
          ),
          'options': list(_PERMISSION_OPTIONS),
        },
      })
      response = await future
    finally:
      turn.permission = None
      self._pending.pop(request_id, None)
    if turn.cancel_requested or session.failed:
      # The interrupt supersedes the approval, or there is no worker to tell.
      return
    approved = False
    result = _object(response.get('result'))
    outcome = _object(result.get('outcome')) if result is not None else None
    if outcome is not None and outcome.get('outcome') == 'selected':
      approved = outcome.get('optionId') == 'allow'
    session.client.send(
      agent_pb2.ClientMessage(
        session_id=session.session_id,
        message_id=uuid.uuid4().hex,
        hitl_response=agent_pb2.HITLResponse(
          call_id=call.id,
          approved=approved,
          source=agent_pb2.APPROVAL_SOURCE_MANUAL,
        ),
      )
    )

  async def _notify(self, method: str, params: JSON) -> None:
    await self._write({'jsonrpc': '2.0', 'method': method, 'params': params})

  async def _write_error(
    self, request_id: object, code: int, message: str
  ) -> None:
    await self._write({
      'jsonrpc': '2.0',
      'id': request_id,
      'error': {'code': code, 'message': message},
    })

  async def _write(self, message: JSON) -> None:
    assert self._output is not None
    data = json.dumps(message, ensure_ascii=False).encode('utf-8') + b'\n'
    async with self._write_lock:
      self._output.write(data)
      self._output.flush()


_DEBUG = bool(os.environ.get('MIRROR_ACP_DEBUG'))


def _debug_event(event: agent_pb2.SessionMessage) -> None:
  """Trace one session message on stderr (set MIRROR_ACP_DEBUG=1)."""
  payload = event.WhichOneof('payload')
  detail = ''
  if payload == 'runtime_delta':
    detail = event.runtime_delta.WhichOneof('payload') or ''
  elif payload == 'agent_message':
    record = event.agent_message
    detail = record.WhichOneof('payload') or ''
    if detail == 'chat':
      kind = record.chat.WhichOneof('message') or ''
      detail += f'/{kind}'
      if kind == 'assistant':
        assistant = record.chat.assistant
        detail += (
          f' content={len(assistant.content)}'
          f' tool_calls={len(assistant.tool_calls)}'
          f' success={record.chat.success}'
        )
    elif detail == 'log':
      detail += f' {record.log.level}: {record.log.text[:160]!r}'
  print(f'mirror acp: event {payload} {detail}', file=sys.stderr)


def _record_user_message(
  event: agent_pb2.SessionMessage,
  conversation_id: str,
  workspace: pathlib.Path,
  model: str,
) -> None:
  """Keep committed user messages visible to `mirror --continue`."""
  message = event.agent_message
  if (
    message.WhichOneof('payload') != 'chat'
    or message.chat.WhichOneof('message') != 'user'
    or message.chat.HasField('display')
    or not message.chat.user.content.strip()
  ):
    return
  with cli_store.Store() as store:
    store.record_user_message(
      conversation_id,
      entry_id=message.id,
      sequence=message.sequence,
      timestamp=message.timestamp.ToDatetime(tzinfo=datetime.UTC)
      if message.HasField('timestamp')
      else datetime.datetime.now(datetime.UTC),
      content=message.chat.user.content,
      cwd=str(workspace),
      model=model,
    )


async def _stdin_reader() -> asyncio.StreamReader:
  reader = asyncio.StreamReader(limit=_MAX_LINE_BYTES)
  protocol = asyncio.StreamReaderProtocol(reader)
  await asyncio.get_running_loop().connect_read_pipe(
    lambda: protocol, sys.stdin.buffer
  )
  return reader


def _consent(workspace: pathlib.Path) -> None:
  """Require the directory's notices to have been accepted interactively."""
  onboarding.ensure_consent(
    workspace, confirm=noninteractive.require_interactive_consent
  )


def run(
  worker_command: Sequence[str],
  worker_config: WorkerConfigFactory,
  *,
  reasoning_effort: str,
) -> int:
  """Serve ACP on stdin/stdout until the client disconnects."""
  server = Server(
    worker_config=worker_config,
    transport=_local_transport(worker_command),
    reasoning_effort=reasoning_effort,
    consent=_consent,
  )

  async def main() -> None:
    await server.serve(await _stdin_reader(), sys.stdout.buffer)

  try:
    asyncio.run(main())
  except KeyboardInterrupt:
    return 130
  return 0
