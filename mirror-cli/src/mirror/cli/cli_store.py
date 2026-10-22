"""CLI-owned conversation metadata and prompt history."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import importlib.resources
import itertools
import os
import pathlib
import sqlite3
from collections.abc import Callable, Sequence
from typing import Self

from google.protobuf import message as protobuf_message
from mirror.protocol import (
  client_pb2,
  mirror_pb2,
)

from mirror import workspace


class SessionOpenError(RuntimeError):
  """A session cannot be opened as requested."""


class DatabaseCompatibilityError(SessionOpenError):
  """The database needs a different client; leave it intact."""


class ConversationNotFoundError(SessionOpenError):
  """The requested conversation does not exist."""


class WorkspaceMismatchError(SessionOpenError):
  """A conversation was resumed from a different project directory."""


@dataclasses.dataclass(frozen=True)
class ConversationInfo:
  """Details shown when choosing a saved conversation."""

  id: str
  title: str | None
  cwd: str
  model: str
  created_at: datetime.datetime
  updated_at: datetime.datetime


@dataclasses.dataclass(frozen=True)
class UserMessage:
  """One canonical user message projected into CLI history."""

  entry_id: str
  sequence: int
  timestamp: datetime.datetime
  content: str


def default_database_path() -> pathlib.Path:
  """Where durable CLI state is stored when no path is given."""
  if path := os.environ.get('MIRROR_CLI_DB'):
    return pathlib.Path(path)
  return pathlib.Path.home() / '.mirror' / 'cli.db'


@dataclasses.dataclass(frozen=True)
class WorktreeOverlay:
  """Optional local metadata projected onto a discovered worktree."""

  alias: str | None = None
  preferences: client_pb2.WorkspacePreferences = dataclasses.field(
    default_factory=client_pb2.WorkspacePreferences
  )

  conversation_id: str | None = None
  conversation_cwd: str | None = None

  @property
  def tag_id(self) -> str | None:
    """Return the assigned tag, with absent proto fields treated as untagged."""
    return (
      self.preferences.tag_id if self.preferences.HasField('tag_id') else None
    )


# user_version is the minimum compatible reader, not an additive-feature counter.
_MIGRATIONS = ('0001_initial.sql',)
_SCHEMA_VERSION = len(_MIGRATIONS)


def _call[T](
  path: str | pathlib.Path | None, operation: Callable[[Store], T]
) -> T:
  with Store(path) as store:
    return operation(store)


async def call[T](
  path: str | pathlib.Path | None, operation: Callable[[Store], T]
) -> T:
  """Run with a thread-owned connection; cancellation waits for its completion."""
  task = asyncio.create_task(asyncio.to_thread(_call, path, operation))
  try:
    return await asyncio.shield(task)
  except asyncio.CancelledError:
    # A write must finish before a session can close or start its replacement.
    while not task.done():
      try:
        await asyncio.shield(task)
      except asyncio.CancelledError:
        continue
    task.result()
    raise


class Store:
  """CLI-owned conversation metadata and prompt history."""

  def __init__(
    self,
    path: str | pathlib.Path | None = None,
    *,
    import_from: str | pathlib.Path | None = None,
  ) -> None:
    self.path = (
      pathlib.Path(path) if path is not None else default_database_path()
    )
    self.path.parent.mkdir(parents=True, exist_ok=True)
    if self.path != pathlib.Path(':memory:'):
      # SQLite copies the database mode to new journal/WAL/SHM files.
      os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
    self._db = sqlite3.connect(self.path, timeout=5)
    self._db.row_factory = sqlite3.Row
    self._db.execute('PRAGMA foreign_keys = ON')
    self._db.execute('PRAGMA busy_timeout = 5000')
    try:
      self._recover_workspace_preview()
      self._prepare_schema(
        pathlib.Path(import_from) if import_from is not None else None
      )
      self._prepare_workspaces()
    except BaseException:
      self._db.close()
      raise

  def _prepare_workspaces(self) -> None:
    columns = {
      row['name']
      for row in self._db.execute('PRAGMA table_info(cli_worktree_overlays)')
    }
    preview = self._db.execute(
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name IN "
      "('cli_workspaces', 'cli_folder_workspaces', 'cli_closed_workspaces')"
    ).fetchone()
    if {
      'path',
      'name',
      'created_at',
      'closed_at',
      'sandbox',
      'conversation_id',
      'conversation_cwd',
    } <= columns and preview is None:
      return
    # Recheck under the lock: another CLI may have added the columns meanwhile.
    self._db.execute('BEGIN IMMEDIATE')
    try:
      self._upgrade_workspace_metadata()
      self._db.commit()
    except BaseException:
      self._db.rollback()
      raise

  def _upgrade_workspace_metadata(self) -> None:
    """Extend existing records and absorb earlier preview tables atomically."""
    columns = {
      row['name']
      for row in self._db.execute('PRAGMA table_info(cli_worktree_overlays)')
    }
    for column in (
      'path',
      'name',
      'created_at',
      'closed_at',
      'sandbox',
      'conversation_id',
      'conversation_cwd',
    ):
      if column not in columns:
        self._db.execute(
          f'ALTER TABLE cli_worktree_overlays ADD COLUMN {column} TEXT'
        )
    tables = {
      row['name']
      for row in self._db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
      )
    }
    for table, fields in (
      ('cli_workspaces', ('path', 'name', 'created_at', 'closed_at')),
      ('cli_folder_workspaces', ('path', 'name', 'created_at')),
      ('cli_closed_workspaces', ('closed_at',)),
    ):
      if table not in tables:
        continue
      actual = tuple(
        row['name'] for row in self._db.execute(f'PRAGMA table_info({table})')
      )
      if actual != ('repository_key', 'id', *fields):
        raise DatabaseCompatibilityError(
          f'Unrecognized workspace preview table {table!r}; database left unchanged.'
        )
      names = ', '.join(fields)
      updates = ', '.join(
        f'{field} = COALESCE(cli_worktree_overlays.{field}, excluded.{field})'
        for field in fields
      )
      self._db.execute(
        f'INSERT INTO cli_worktree_overlays(repository_key, id, preferences, {names}) '
        f"SELECT repository_key, id, X'', {names} FROM {table} WHERE true "
        f'ON CONFLICT(repository_key, id) DO UPDATE SET {updates}'
      )
      self._db.execute(f'DROP TABLE {table}')

  def _recover_workspace_preview(self) -> None:
    """Repair only the two additive schemas shipped by the folder-workspace preview."""
    if self._db.execute('PRAGMA user_version').fetchone()[0] != 2:
      return
    expected: dict[str, tuple[str, ...]] = {
      'cli_user_messages': (
        'conversation_id',
        'entry_id',
        'sequence',
        'timestamp',
        'content',
        'cwd',
        'git_key',
        'model',
      ),
      'cli_worktree_overlays': ('repository_key', 'id', 'alias', 'preferences'),
      'cli_workspaces': (
        'repository_key',
        'id',
        'path',
        'name',
        'created_at',
        'closed_at',
      ),
      'cli_folder_workspaces': (
        'repository_key',
        'id',
        'path',
        'name',
        'created_at',
      ),
      'cli_closed_workspaces': ('repository_key', 'id', 'closed_at'),
    }
    self._db.execute('BEGIN IMMEDIATE')
    try:
      columns = {
        table: tuple(
          row['name'] for row in self._db.execute(f'PRAGMA table_info({table})')
        )
        for table in expected
      }
      extended_overlay = (
        *expected['cli_worktree_overlays'],
        'path',
        'name',
        'created_at',
        'closed_at',
      )
      # Recognize only known additive columns left by a newer reader.
      if columns['cli_worktree_overlays'] in (
        extended_overlay,
        (*extended_overlay, 'sandbox'),
        (*extended_overlay, 'sandbox', 'conversation_id', 'conversation_cwd'),
      ):
        expected['cli_worktree_overlays'] = columns['cli_worktree_overlays']
      if (
        columns['cli_user_messages'] != expected['cli_user_messages']
        or columns['cli_worktree_overlays'] != expected['cli_worktree_overlays']
        or not (columns['cli_workspaces'] or columns['cli_folder_workspaces'])
        or any(
          value and value != expected[table] for table, value in columns.items()
        )
        or bool(columns['cli_folder_workspaces'])
        != bool(columns['cli_closed_workspaces'])
      ):
        raise DatabaseCompatibilityError(
          'Unrecognized CLI schema version 2; database left unchanged. Use a compatible Mirror version.'
        )
      self._upgrade_workspace_metadata()
      self._db.execute('PRAGMA user_version = 1')
      self._db.commit()
    except BaseException:
      self._db.rollback()
      raise

  def _prepare_schema(self, import_from: pathlib.Path | None) -> None:
    version = int(self._db.execute('PRAGMA user_version').fetchone()[0])
    if version > _SCHEMA_VERSION:
      raise DatabaseCompatibilityError(
        f'CLI database schema version {version} is newer than this binary. '
        'Use a compatible Mirror version; the database was left unchanged.'
      )
    if version == _SCHEMA_VERSION:
      return

    attached = False
    imports: list[str] = []
    if (
      version == 0
      and import_from is not None
      and import_from.exists()
      and import_from.resolve() != self.path.resolve()
    ):
      self._db.execute('ATTACH DATABASE ? AS legacy', (str(import_from),))
      attached = True
      tables = {
        str(row['name'])
        for row in self._db.execute(
          "SELECT name FROM legacy.sqlite_master WHERE type = 'table'"
        )
      }
      if 'cli_user_messages' in tables:
        imports.append(
          """
          INSERT OR IGNORE INTO cli_user_messages(
            conversation_id, entry_id, sequence, timestamp, content,
            cwd, git_key, model
          )
          SELECT conversation_id, entry_id, sequence, timestamp, content,
                 cwd, git_key, model
          FROM legacy.cli_user_messages;
          """
        )
      if 'cli_worktree_overlays' in tables:
        imports.append(
          """
          INSERT OR IGNORE INTO cli_worktree_overlays(
            repository_key, id, alias, preferences
          )
          SELECT repository_key, id, alias, preferences
          FROM legacy.cli_worktree_overlays;
          """
        )
      if 'cli_worktree_aliases' in tables:
        columns = {
          str(row['name'])
          for row in self._db.execute(
            'PRAGMA legacy.table_info(cli_worktree_aliases)'
          )
        }
        id_column = (
          'worktree_path' if 'worktree_path' in columns else 'worktree_id'
        )
        imports.append(
          f"""
          INSERT OR IGNORE INTO cli_worktree_overlays(
            repository_key, id, alias, preferences
          )
          SELECT repository_key, {id_column}, alias, X''
          FROM legacy.cli_worktree_aliases;
          """
        )
      if {
        'cli_conversation_names',
        'cli_user_messages',
      }.issubset(tables):
        imports.append(
          """
          INSERT OR IGNORE INTO cli_worktree_overlays(
            repository_key, id, alias, preferences
          )
          SELECT COALESCE(messages.git_key, messages.cwd),
                 names.conversation_id, names.title, X''
          FROM legacy.cli_conversation_names AS names
          JOIN legacy.cli_user_messages AS messages
            ON messages.conversation_id = names.conversation_id
          WHERE messages.timestamp = (
            SELECT MAX(latest.timestamp)
            FROM legacy.cli_user_messages AS latest
            WHERE latest.conversation_id = names.conversation_id
          );
          """
        )

    script = ['BEGIN IMMEDIATE;']
    for migration_version, filename in enumerate(_MIGRATIONS, start=1):
      if migration_version <= version:
        continue
      script.append(
        importlib.resources
        .files(__package__)
        .joinpath('migrations', filename)
        .read_text()
      )
      script.append(f'PRAGMA user_version = {migration_version};')
    script.extend(imports)
    script.append('COMMIT;')
    try:
      self._db.executescript('\n'.join(script))
    except BaseException:
      if self._db.in_transaction:
        self._db.rollback()
      raise
    finally:
      if attached:
        self._db.execute('DETACH DATABASE legacy')

  def __enter__(self) -> Self:
    return self

  def __exit__(self, *_error: object) -> None:
    self._db.close()

  def saved_workspaces(
    self, repository_key: str
  ) -> list[tuple[str, str, str, str | None]]:
    """Read explicitly opened workspaces in insertion order.

    Each is `(id, path, name, sandbox)`, where `sandbox` is the JSON a sandbox
    workspace was saved with.
    """
    return [
      (
        str(row['id']),
        str(row['path']),
        str(row['name']),
        None if row['sandbox'] is None else str(row['sandbox']),
      )
      for row in self._db.execute(
        'SELECT id, path, name, sandbox FROM cli_worktree_overlays '
        'WHERE repository_key = ? AND closed_at IS NULL AND path IS NOT NULL ORDER BY rowid',
        (repository_key,),
      )
    ]

  def save_workspace(
    self,
    repository_key: str,
    workspace_id: str,
    path: str,
    name: str,
    sandbox: str | None = None,
  ) -> None:
    """Persist a workspace handle; duplicate paths remain independent."""
    with self._db:
      self._db.execute(
        'INSERT INTO cli_worktree_overlays(repository_key, id, path, name, sandbox, preferences, created_at) '
        "VALUES (?, ?, ?, ?, ?, X'', CURRENT_TIMESTAMP) "
        'ON CONFLICT(repository_key, id) DO UPDATE SET '
        'path = excluded.path, name = excluded.name, sandbox = excluded.sandbox, closed_at = NULL, '
        'created_at = COALESCE(cli_worktree_overlays.created_at, excluded.created_at)',
        (repository_key, workspace_id, path, name, sandbox),
      )

  def closed_workspaces(self, repository_key: str) -> set[str]:
    """Return workspaces explicitly hidden from automatic discovery."""
    return {
      str(row['id'])
      for row in self._db.execute(
        'SELECT id FROM cli_worktree_overlays WHERE repository_key = ? AND closed_at IS NOT NULL',
        (repository_key,),
      )
    }

  def close_workspace(self, repository_key: str, workspace_id: str) -> None:
    """Close a roster entry, never its directory or conversation history."""
    with self._db:
      self._db.execute(
        'INSERT INTO cli_worktree_overlays(repository_key, id, preferences, closed_at) '
        "VALUES (?, ?, X'', CURRENT_TIMESTAMP) "
        'ON CONFLICT(repository_key, id) DO UPDATE SET closed_at = excluded.closed_at',
        (repository_key, workspace_id),
      )

  def reopen_workspace(self, repository_key: str, workspace_id: str) -> None:
    """Return a closed roster entry to automatic discovery."""
    with self._db:
      self._db.execute(
        'UPDATE cli_worktree_overlays SET closed_at = NULL '
        'WHERE repository_key = ? AND id = ?',
        (repository_key, workspace_id),
      )

  def record_user_message(
    self,
    conversation_id: str,
    *,
    entry_id: str,
    sequence: int,
    timestamp: datetime.datetime,
    content: str,
    cwd: str,
    model: str,
  ) -> None:
    """Store one user message and its current conversation metadata."""
    self.record_user_messages(
      conversation_id,
      [UserMessage(entry_id, sequence, timestamp, content)],
      cwd=cwd,
      model=model,
    )

  def record_user_messages(
    self,
    conversation_id: str,
    messages: Sequence[UserMessage],
    *,
    cwd: str,
    model: str,
  ) -> None:
    """Store user messages and conversation metadata in one transaction."""
    self._validate_id(conversation_id)
    resolved_cwd = workspace.resolve_cwd(cwd)
    git_key = workspace.git_workspace_key(resolved_cwd)
    with self._db:
      self._db.executemany(
        """
        INSERT INTO cli_user_messages(
          conversation_id, entry_id, sequence, timestamp, content,
          cwd, git_key, model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(conversation_id, entry_id) DO UPDATE SET
          sequence = excluded.sequence,
          timestamp = excluded.timestamp,
          content = excluded.content,
          cwd = excluded.cwd,
          git_key = excluded.git_key,
          model = excluded.model
        """,
        (
          (
            conversation_id,
            message.entry_id,
            message.sequence,
            message.timestamp.isoformat(),
            message.content,
            resolved_cwd,
            git_key,
            model,
          )
          for message in messages
        ),
      )

  def validate_workspace(
    self, conversation_id: str, cwd: str
  ) -> ConversationInfo:
    """Return conversation details if it belongs to `cwd`."""
    info = self.get_info(conversation_id)
    resolved_cwd = workspace.resolve_cwd(cwd)
    if info.cwd != resolved_cwd:
      raise WorkspaceMismatchError(
        f'conversation {conversation_id!r} was started in {info.cwd}; '
        f'the current directory is {resolved_cwd}'
      )
    return info

  def list_infos(
    self,
    *,
    cwd: str | None = None,
    scope: workspace.SessionScope = workspace.SessionScope.CWD,
  ) -> list[ConversationInfo]:
    """List recent conversations in the selected scope."""
    records = self._infos()
    if cwd is None or scope is workspace.SessionScope.ALL:
      return [info for info, _ in records]
    resolved_cwd = workspace.resolve_cwd(cwd)
    if scope is workspace.SessionScope.CWD:
      return [info for info, _ in records if info.cwd == resolved_cwd]
    git_key = workspace.git_workspace_key(resolved_cwd)
    if git_key is None:
      return [info for info, _ in records if info.cwd == resolved_cwd]
    return [
      info
      for info, stored_git_key in records
      if stored_git_key == git_key
      or (
        stored_git_key is None
        and workspace.git_workspace_key(info.cwd) == git_key
      )
    ]

  def latest_by_cwd(self) -> dict[str, ConversationInfo]:
    """Read the latest conversation per directory in one metadata snapshot."""
    latest: dict[str, ConversationInfo] = {}
    for info, _ in self._infos():
      latest.setdefault(info.cwd, info)
    return latest

  def latest(self, *, cwd: str) -> ConversationInfo | None:
    """Return the most recently updated conversation in `cwd`."""
    infos = self.list_infos(cwd=cwd, scope=workspace.SessionScope.CWD)
    return infos[0] if infos else None

  def find_by_reference(self, reference: str) -> list[ConversationInfo]:
    """Match a full conversation id or id prefix."""
    infos = self.list_infos(scope=workspace.SessionScope.ALL)
    exact = [info for info in infos if info.id == reference]
    return exact or [info for info in infos if info.id.startswith(reference)]

  def get_info(self, conversation_id: str) -> ConversationInfo:
    """Return metadata for one conversation."""
    self._validate_id(conversation_id)
    for info, _ in self._infos():
      if info.id == conversation_id:
        return info
    raise ConversationNotFoundError(
      f'conversation {conversation_id!r} does not exist'
    )

  @staticmethod
  def _preferences(value: bytes) -> client_pb2.WorkspacePreferences:
    """Decode stored preferences, moving pre-`definition` fields into it.

    `AgentDefinition` reuses the field numbers those fields had, so decoding
    old bytes as a definition recovers them.
    """
    try:
      preferences = client_pb2.WorkspacePreferences.FromString(bytes(value))
      if not preferences.HasField('definition'):
        legacy = mirror_pb2.AgentDefinition.FromString(bytes(value))
        legacy.DiscardUnknownFields()
        if legacy.ListFields():
          preferences.definition.CopyFrom(legacy)
    except (protobuf_message.DecodeError, TypeError) as error:
      raise ValueError(
        'Cannot restore corrupt workspace configuration'
      ) from error
    return preferences

  def worktree_overlays(
    self, repository_key: str
  ) -> dict[str, WorktreeOverlay]:
    """Return optional metadata for worktrees in one repository."""
    rows = self._db.execute(
      """
      SELECT id, alias, preferences, conversation_id, conversation_cwd
      FROM cli_worktree_overlays
      WHERE repository_key = ?
      """,
      (repository_key,),
    )
    overlays: dict[str, WorktreeOverlay] = {}
    for row in rows:
      overlays[str(row['id'])] = WorktreeOverlay(
        alias=str(row['alias']) if row['alias'] is not None else None,
        preferences=self._preferences(row['preferences']),
        conversation_id=str(row['conversation_id'])
        if row['conversation_id'] is not None
        else None,
        conversation_cwd=str(row['conversation_cwd'])
        if row['conversation_cwd'] is not None
        else None,
      )
    return overlays

  def delete_workspace_tag(self, tag_id: str) -> None:
    """Clear a deleted definition's assignments, including closed workspaces."""
    with self._db:
      self._db.execute('BEGIN IMMEDIATE')
      rows = self._db.execute(
        'SELECT repository_key, id, preferences FROM cli_worktree_overlays'
      ).fetchall()
      for row in rows:
        preferences = self._preferences(row['preferences'])
        if preferences.HasField('tag_id') and preferences.tag_id == tag_id:
          preferences.ClearField('tag_id')
          self._db.execute(
            'UPDATE cli_worktree_overlays SET preferences = ? WHERE repository_key = ? AND id = ?',
            (
              preferences.SerializeToString(deterministic=True),
              row['repository_key'],
              row['id'],
            ),
          )

  def set_workspace_tag(
    self, repository_key: str, workspace_id: str, tag_id: str | None
  ) -> None:
    """Update only the assigned tag in workspace preferences."""
    with self._db:
      self._db.execute('BEGIN IMMEDIATE')
      row = self._db.execute(
        'SELECT preferences FROM cli_worktree_overlays WHERE repository_key = ? AND id = ?',
        (repository_key, workspace_id),
      ).fetchone()
      preferences = (
        self._preferences(row['preferences'])
        if row is not None
        else client_pb2.WorkspacePreferences()
      )
      if tag_id is None:
        preferences.ClearField('tag_id')
      else:
        preferences.tag_id = tag_id
      self.set_worktree_preferences(repository_key, workspace_id, preferences)

  def set_worktree_alias(
    self, repository_key: str, worktree_id: str, alias: str
  ) -> str:
    """Set and return a normalized alias unique within one repository."""
    alias = ' '.join(alias.split())
    if not alias:
      raise ValueError('worktree alias must not be empty')
    if len(alias) > 80:
      alias = alias[:79] + '…'
    try:
      with self._db:
        self._db.execute(
          """
          INSERT INTO cli_worktree_overlays(
            repository_key, id, alias, preferences
          ) VALUES (?, ?, ?, ?)
          ON CONFLICT(repository_key, id) DO UPDATE SET alias = excluded.alias
          """,
          (repository_key, worktree_id, alias, b''),
        )
    except sqlite3.IntegrityError as error:
      raise ValueError(f'worktree alias {alias!r} is already in use') from error
    return alias

  def set_worktree_preferences(
    self,
    repository_key: str,
    worktree_id: str,
    preferences: client_pb2.WorkspacePreferences,
  ) -> None:
    """Persist workspace preferences without changing its alias."""
    with self._db:
      self._db.execute(
        """
        INSERT INTO cli_worktree_overlays(
          repository_key, id, alias, preferences
        ) VALUES (?, ?, NULL, ?)
        ON CONFLICT(repository_key, id)
        DO UPDATE SET preferences = excluded.preferences
        """,
        (
          repository_key,
          worktree_id,
          preferences.SerializeToString(deterministic=True),
        ),
      )

  def set_workspace_conversation(
    self,
    repository_key: str,
    workspace_id: str,
    conversation_id: str,
    cwd: pathlib.Path,
  ) -> None:
    """Remember a tab's conversation without changing its launch preferences."""
    self._validate_id(conversation_id)
    with self._db:
      self._db.execute(
        """
        INSERT INTO cli_worktree_overlays(
          repository_key, id, preferences, conversation_id, conversation_cwd
        ) VALUES (?, ?, X'', ?, ?)
        ON CONFLICT(repository_key, id)
        DO UPDATE SET conversation_id = excluded.conversation_id,
                      conversation_cwd = excluded.conversation_cwd
        """,
        (repository_key, workspace_id, conversation_id, str(cwd.resolve())),
      )

  def delete_worktree_overlay(
    self, repository_key: str, worktree_id: str
  ) -> None:
    """Forget local metadata for an explicitly deleted worktree."""
    with self._db:
      self._db.execute(
        """
        DELETE FROM cli_worktree_overlays
        WHERE repository_key = ? AND id = ?
        """,
        (repository_key, worktree_id),
      )

  def move_worktree_overlay(
    self, repository_key: str, source_id: str, destination_id: str
  ) -> None:
    """Move legacy location-keyed metadata to a discovered identity."""
    with self._db:
      self._db.execute(
        """
        UPDATE OR IGNORE cli_worktree_overlays SET id = ?
        WHERE repository_key = ? AND id = ?
        """,
        (destination_id, repository_key, source_id),
      )

  def rewind(self, conversation_id: str, entry_id: str) -> None:
    """Remove projected user messages from `entry_id` onward."""
    self._validate_id(conversation_id)
    with self._db:
      sequence = self._db.execute(
        """
        SELECT sequence FROM cli_user_messages
        WHERE conversation_id = ? AND entry_id = ?
        """,
        (conversation_id, entry_id),
      ).fetchone()
      if sequence is None:
        return
      self._db.execute(
        """
        DELETE FROM cli_user_messages
        WHERE conversation_id = ? AND sequence >= ?
        """,
        (conversation_id, sequence['sequence']),
      )

  def recent_user_messages(
    self, conversation_id: str | None = None, limit: int = 1000
  ) -> list[str]:
    """Return projected user prompts, newest first."""
    rows = self._db.execute(
      """
      SELECT content FROM cli_user_messages
      ORDER BY (conversation_id != ?), timestamp DESC
      LIMIT ?
      """,
      (conversation_id, limit),
    )
    texts: list[str] = []
    for row in rows:
      content = str(row['content'])
      if not content.strip() or (texts and texts[-1] == content):
        continue
      texts.append(content)
    return texts

  @staticmethod
  def _title(content: str) -> str:
    compact = ' '.join(content.split())
    return compact if len(compact) <= 80 else compact[:79] + '…'

  @staticmethod
  def _validate_id(conversation_id: str) -> None:
    if not conversation_id or not all(
      character.isascii() and (character.isalnum() or character in '._-')
      for character in conversation_id
    ):
      raise ValueError(
        f'conversation id {conversation_id!r} must match ^[A-Za-z0-9._-]+$'
      )

  def _infos(self) -> list[tuple[ConversationInfo, str | None]]:
    messages = self._db.execute(
      """
      SELECT * FROM cli_user_messages
      ORDER BY conversation_id, timestamp, sequence
      """
    ).fetchall()
    records: list[tuple[ConversationInfo, str | None]] = []
    for conversation_id, rows in itertools.groupby(
      messages, key=lambda row: str(row['conversation_id'])
    ):
      conversation = list(rows)
      first = conversation[0]
      latest = conversation[-1]
      title = str(first['content'])
      records.append((
        ConversationInfo(
          id=conversation_id,
          title=self._title(title),
          cwd=str(latest['cwd']),
          model=str(latest['model']),
          created_at=datetime.datetime.fromisoformat(first['timestamp']),
          updated_at=datetime.datetime.fromisoformat(latest['timestamp']),
        ),
        latest['git_key'],
      ))
    records.sort(key=lambda record: record[0].updated_at, reverse=True)
    return records
