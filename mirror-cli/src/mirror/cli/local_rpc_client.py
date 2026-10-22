"""Local AgentRPC worker transport."""

from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import json
import os
import pathlib
import signal
import sys
from collections.abc import Mapping, Sequence

import grpc.aio
from google.protobuf import message
from mirror.protocol import agent_pb2

from mirror.cli import host_tools as _host_tools
from mirror.cli import rpc_client

_WORKER_LAUNCHER = (
  'import os,sys; '
  'os.chdir(sys.argv[1]); '
  'os.setsid(); '
  'os.execvp(sys.argv[2], sys.argv[2:])'
)


async def _signal_group(
  process: asyncio.subprocess.Process, sig: signal.Signals
) -> None:
  """Signal the worker's group, tolerating an already-exited worker."""
  try:
    os.killpg(process.pid, sig)
    return
  except ProcessLookupError:
    return
  except PermissionError as error:
    # macOS reports EPERM for a group containing only an unreaped zombie.
    try:
      await asyncio.wait_for(process.wait(), timeout=3)
    except TimeoutError:
      raise error from None
  # Retry after reaping: surviving descendants must still receive the signal.
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    pass


class Transport:
  """Start a local worker process and own its AgentRPC channel."""

  def __init__(
    self,
    worker_command: Sequence[str],
    worker_config: Mapping[str, object],
    *,
    workspace: pathlib.Path,
    log_dir: pathlib.Path,
    on_message: rpc_client.MessageHandler | None = None,
  ) -> None:
    self._worker_command = list(worker_command)
    self._worker_config = dict(worker_config)
    self._workspace = workspace
    self._log_dir = log_dir
    self._on_message = on_message
    self._process: asyncio.subprocess.Process | None = None
    self._channel: grpc.aio.Channel | None = None
    self._log_tasks: list[asyncio.Task[None]] = []

  @property
  def process(self) -> asyncio.subprocess.Process | None:
    """The active worker process."""
    return self._process

  async def connect(
    self, host_tools: Sequence[_host_tools.ToolDefinition] = ()
  ) -> grpc.aio.Channel:
    """Start the worker and return its ready AgentRPC channel."""
    address = '127.0.0.1:0'
    config = self._worker_config | {
      'rpc_address': address,
      'exit_on_stdin_close': True,
      'host_tools': [
        {
          key: value
          for key, value in dataclasses.asdict(tool).items()
          if key != 'global_'
        }
        for tool in host_tools
      ],
    }
    self._log_dir.mkdir(parents=True, exist_ok=True)
    try:
      self._process = await asyncio.create_subprocess_exec(
        str(pathlib.Path(sys.executable).resolve()),
        '-c',
        _WORKER_LAUNCHER,
        str(self._workspace),
        *self._worker_command,
        json.dumps(config),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=False,
      )
      process = self._process
      self._log_tasks = [
        asyncio.create_task(
          self._collect_logs(process.stderr, self._log_dir / 'stderr')
        ),
      ]
      try:
        address = await asyncio.wait_for(
          self._read_worker_address(process), timeout=30
        )
      except TimeoutError as error:
        raise RuntimeError(
          'agent worker did not report its address within 30 seconds; '
          f'see {self._log_dir / "stderr"}'
        ) from error
      self._log_tasks.append(
        asyncio.create_task(
          self._collect_logs(process.stdout, self._log_dir / 'stdout')
        )
      )
      self._channel = grpc.aio.insecure_channel(
        address,
        # This channel only talks to our child; external traffic keeps its proxy.
        options=rpc_client.GRPC_CHANNEL_OPTIONS
        + (('grpc.enable_http_proxy', 0),),
      )
      ready = asyncio.create_task(self._channel.channel_ready())
      exited = asyncio.create_task(process.wait())
      done, _ = await asyncio.wait(
        (ready, exited), return_when=asyncio.FIRST_COMPLETED
      )
      if exited in done:
        ready.cancel()
        await asyncio.gather(ready, return_exceptions=True)
        raise RuntimeError(
          f'agent worker exited with status {process.returncode}; '
          f'see {self._log_dir / "stderr"}'
        )
      exited.cancel()
      await asyncio.gather(exited, return_exceptions=True)
      return self._channel
    except BaseException:
      await self.close()
      raise

  async def close(self) -> None:
    """Close the channel, stop the worker, and await its log readers."""
    if self._channel is not None:
      await self._channel.close(grace=None)
      self._channel = None
    process = self._process
    if process is not None and process.returncode is None:
      # Let the worker drain its children before forced process-group cleanup.
      try:
        process.terminate()
      except ProcessLookupError:
        pass
      try:
        await asyncio.wait_for(process.wait(), timeout=3)
      except TimeoutError:
        pass
      await _signal_group(process, signal.SIGKILL)
      await process.wait()
    if self._log_tasks:
      _, pending = await asyncio.wait(self._log_tasks, timeout=1)
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
    self._log_tasks.clear()
    self._process = None

  async def _read_worker_address(
    self, process: asyncio.subprocess.Process
  ) -> str:
    """Read the address after the worker has atomically bound its listener."""
    if process.stdout is None:
      raise RuntimeError('agent worker has no stdout pipe')
    address_prefix = b'MIRROR_RPC_ADDRESS='
    error_prefix = b'MIRROR_RPC_ERROR='
    while line := await process.stdout.readline():
      if await self._consume_event(line):
        continue
      if line.startswith(address_prefix):
        return line.removeprefix(address_prefix).decode().strip()
      if line.startswith(error_prefix):
        raise RuntimeError(line.removeprefix(error_prefix).decode().strip())
      await self._append_log(self._log_dir / 'stdout', line)
    status = await process.wait()
    raise RuntimeError(
      f'agent worker exited with status {status} before reporting its address; '
      f'see {self._log_dir / "stderr"}'
    )

  async def _consume_event(self, line: bytes) -> bool:
    prefix = b'MIRROR_SESSION_MESSAGE='
    if not line.startswith(prefix):
      return False
    try:
      event = agent_pb2.SessionMessage.FromString(
        base64.b64decode(line.removeprefix(prefix).strip(), validate=True)
      )
    except (binascii.Error, message.DecodeError):
      raise RuntimeError('Invalid worker event') from None
    if self._on_message is not None:
      await rpc_client.deliver_message(self._on_message, event)
    return True

  @staticmethod
  async def _append_log(path: pathlib.Path, content: bytes) -> None:
    def write() -> None:
      descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
      with os.fdopen(descriptor, 'ab') as sink:
        sink.write(content)

    writing = asyncio.create_task(asyncio.to_thread(write))
    try:
      await asyncio.shield(writing)
    except asyncio.CancelledError:
      # Closing the transport must not leave a log write running in its thread.
      while not writing.done():
        try:
          await asyncio.shield(writing)
        except asyncio.CancelledError:
          continue
      writing.result()
      raise

  async def _collect_logs(
    self, stream: asyncio.StreamReader | None, path: pathlib.Path
  ) -> None:
    if stream is None:
      return
    try:
      await self._append_log(path, b'')
      while chunk := await (
        stream.readline() if path.name == 'stdout' else stream.read(1 << 16)
      ):
        if path.name == 'stdout' and await self._consume_event(chunk):
          continue
        # Only stdout has protocol framing; batch ordinary diagnostics by bytes.
        await self._append_log(path, chunk)
    except Exception:  # noqa: BLE001 - never let worker logging block.
      while await stream.read(1 << 16):
        await asyncio.sleep(0)
