"""Discover and create isolated Git worktrees or Jujutsu workspaces."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence

from mirror import project

_GIT = shutil.which('git') or 'git'
_JJ = shutil.which('jj') or 'jj'
_SAFE_WORKTREE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
_JJ_STALE_WORKING_COPY = 'working copy is stale'


class WorktreeError(RuntimeError):
  """An isolated workspace operation failed."""


def _worktree_name(name: str | None) -> str:
  normalized = name.strip() if name is not None else ''
  if not normalized:
    return uuid.uuid4().hex[:8]
  if (
    not _SAFE_WORKTREE_NAME.fullmatch(normalized)
    or '..' in normalized
    or normalized.endswith(('.', '.lock'))
  ):
    raise WorktreeError(
      'worktree name must be 1-64 letters, numbers, dots, dashes, or underscores'
    )
  return normalized


@dataclasses.dataclass(frozen=True)
class WorktreeIdentity:
  """Opaque provider-owned identity independent of worktree location."""

  provider: str
  id: str


@dataclasses.dataclass(frozen=True)
class Worktree:
  """An isolated checkout sharing its repository's history."""

  path: pathlib.Path
  name: str | None
  identity: WorktreeIdentity | None = None
  warning: str | None = dataclasses.field(default=None, compare=False)

  @property
  def provider(self) -> str:
    """The provider namespace that owns `id`."""
    return self.identity.provider if self.identity is not None else 'local'

  @property
  def id(self) -> str:
    """The immutable identifier used by the underlying VCS."""
    if self.identity is not None:
      return self.identity.id
    return self.name or str(self.path.resolve())

  @property
  def label(self) -> str:
    """The compact name shown in the worktree switcher."""
    if self.name is not None and self.name.startswith('mirror/'):
      return self.path.name
    return self.name or self.path.name or str(self.path)


def _creation_time(worktree: Worktree) -> int | None:
  """The timestamp of the stable marker created with the checkout."""
  for marker in (worktree.path / '.jj' / 'repo', worktree.path / '.git'):
    try:
      return marker.lstat().st_mtime_ns
    except FileNotFoundError:
      pass
  try:
    return worktree.path.lstat().st_mtime_ns
  except FileNotFoundError:
    return None


def _order_worktrees(
  worktrees: Sequence[Worktree], default: pathlib.Path | None
) -> tuple[Worktree, ...]:
  """Put the default checkout first, then order siblings by creation."""
  available: list[tuple[Worktree, int]] = []
  for worktree in worktrees:
    created_at = _creation_time(worktree)
    if created_at is not None:
      available.append((worktree, created_at))
  return tuple(
    worktree
    for worktree, _ in sorted(
      available,
      key=lambda item: (
        item[0].path != default,
        item[1],
        str(item[0].path),
      ),
    )
  )


def _git(
  cwd: pathlib.Path,
  arguments: Sequence[str],
  *,
  environment: Mapping[str, str] | None = None,
  timeout: float | None = None,
) -> str:
  completed = subprocess.run(
    (_GIT, '-C', str(cwd), *arguments),
    check=False,
    capture_output=True,
    text=True,
    env=environment,
    close_fds=False,
    timeout=timeout,
  )
  if completed.returncode != 0:
    detail = completed.stderr.strip() or completed.stdout.strip()
    raise WorktreeError(detail or f'git {" ".join(arguments)} failed')
  return completed.stdout.strip()


def _jj(
  cwd: pathlib.Path, arguments: Sequence[str], *, timeout: float | None = None
) -> str:
  try:
    completed = subprocess.run(
      (_JJ, '--repository', str(cwd), *arguments),
      check=False,
      capture_output=True,
      text=True,
      close_fds=False,
      timeout=timeout,
    )
  except FileNotFoundError as error:
    raise WorktreeError('jj is required for this Jujutsu repository') from error
  if completed.returncode != 0:
    detail = completed.stderr.strip() or completed.stdout.strip()
    raise WorktreeError(detail or f'jj {" ".join(arguments)} failed')
  return completed.stdout.strip()


def _list_git_worktrees(root: pathlib.Path) -> tuple[Worktree, ...]:
  output = _git(root, ('worktree', 'list', '--porcelain'))
  worktrees: list[Worktree] = []
  path: pathlib.Path | None = None
  branch: str | None = None
  prunable = False
  for line in (*output.splitlines(), ''):
    if not line:
      if path is not None and not prunable:
        resolved = path.resolve()
        worktrees.append(
          Worktree(
            resolved,
            branch,
            WorktreeIdentity('git', str(resolved)),
          )
        )
      path = None
      branch = None
      prunable = False
    elif line.startswith('worktree '):
      path = pathlib.Path(line.removeprefix('worktree '))
    elif line.startswith('branch refs/heads/'):
      branch = line.removeprefix('branch refs/heads/')
    elif line.startswith('prunable'):
      prunable = True
  default = worktrees[0].path if worktrees else None
  return _order_worktrees(worktrees, default)


def _list_jj_workspaces(root: pathlib.Path) -> tuple[Worktree, ...]:
  fields = _jj(
    root,
    (
      'workspace',
      'list',
      '--ignore-working-copy',
      '--template',
      'name ++ "\\0" ++ root ++ "\\0"',
    ),
  ).split('\0')[:-1]
  worktrees: list[Worktree] = []
  for name, path in zip(fields[::2], fields[1::2], strict=True):
    directory = pathlib.Path(path)
    # Jj embeds errors for forgotten roots in the template output.
    if not directory.is_absolute() or not directory.is_dir():
      continue
    worktrees.append(
      Worktree(directory.resolve(), name, WorktreeIdentity('jj', name))
    )
  default = next(
    (worktree.path for worktree in worktrees if worktree.name == 'default'),
    None,
  )
  return _order_worktrees(worktrees, default)


def current_worktree(cwd: pathlib.Path) -> Worktree:
  """Resolve the launch checkout identity before constructing its agent."""
  root = project.repository_root(cwd.resolve()) or cwd.resolve()
  if (root / '.jj').is_dir():
    # Resolve the identity before applying saved model and tool restrictions.
    fields = (
      _jj(
        root,
        (
          'workspace',
          'list',
          '--ignore-working-copy',
          '--template',
          'name ++ "\\0" ++ root ++ "\\0"',
        ),
      )
      .removesuffix('\0')
      .split('\0')
    )
    # Jj prints missing-root errors inline; keep them separate from identities.
    name = next(
      (
        name
        for name, path in zip(fields[::2], fields[1::2], strict=True)
        if path == str(root)
      ),
      None,
    )
    if not name:
      raise WorktreeError(f'could not identify Jujutsu workspace at {root}')
    return Worktree(root, name, WorktreeIdentity('jj', name))
  if (root / '.git').exists():
    try:
      name = _git(root, ('symbolic-ref', '--quiet', '--short', 'HEAD'))
    except WorktreeError:
      name = None
    return Worktree(root, name, WorktreeIdentity('git', str(root)))
  return Worktree(root, None)


def list_worktrees(cwd: str | pathlib.Path) -> tuple[Worktree, ...]:
  """List the isolated checkouts belonging to the repository containing `cwd`."""
  root = pathlib.Path(cwd).resolve()
  if (jj_root := project.jj_workspace_root(root)) is not None:
    return _list_jj_workspaces(jj_root)
  return _list_git_worktrees(root)


def is_managed(worktree: Worktree) -> bool:
  """Whether `worktree` lives under Mirror's managed checkout directory."""
  managed = (pathlib.Path.home() / '.mirror' / 'worktrees').resolve()
  return worktree.path.resolve().is_relative_to(managed)


def is_managed_directory(path: pathlib.Path) -> bool:
  """Whether the directory is a checkout root under Mirror's managed directory."""
  path = path.resolve()
  managed = (pathlib.Path.home() / '.mirror' / 'worktrees').resolve()
  return path.is_relative_to(managed) and project.repository_root(path) == path


def drop_worktree(worktree: Worktree, *, confirmed: bool = False) -> None:
  """Delete a checkout and unregister its workspace."""
  managed = is_managed(worktree)
  if not managed and not confirmed:
    raise WorktreeError('dropping this worktree requires confirmation')
  if (jj_root := project.jj_workspace_root(worktree.path)) is not None:
    if worktree.name is None:
      raise WorktreeError('Jujutsu workspace has no name')
    if worktree.name == 'default':
      raise WorktreeError('the primary Jujutsu workspace cannot be dropped')
    _jj(
      jj_root,
      ('workspace', 'forget', '--ignore-working-copy', worktree.name),
    )
    shutil.rmtree(worktree.path)
    return
  listed = list_worktrees(worktree.path)
  if listed and listed[0].path == worktree.path.resolve():
    raise WorktreeError('the primary Git worktree cannot be dropped')
  _git(worktree.path, ('worktree', 'remove', '--force', str(worktree.path)))


def _managed_parent(shared_repository: pathlib.Path, name: str) -> pathlib.Path:
  digest = hashlib.sha256(str(shared_repository).encode()).hexdigest()[:8]
  return pathlib.Path.home() / '.mirror' / 'worktrees' / f'{name}-{digest}'


def _default_remote(source: pathlib.Path) -> str | None:
  remotes = _git(source, ('remote',)).splitlines()
  if 'origin' in remotes:
    return 'origin'
  if len(remotes) == 1:
    return remotes[0]
  if not remotes:
    return None
  raise WorktreeError(
    '/worktree cannot choose between multiple remotes without origin. '
    'Configure an origin remote, or create a checkout with Git/Jujutsu '
    'and open it with /folder.'
  )


def _remote_default(source: pathlib.Path, remote: str) -> str:
  """Read the remote's advertised default branch for a scoped Jujutsu fetch."""
  output = _git(
    source, ('ls-remote', '--symref', '--', remote, 'HEAD'), timeout=10
  )
  for line in output.splitlines():
    if line.startswith('ref: refs/heads/') and line.endswith('\tHEAD'):
      return line.removeprefix('ref: refs/heads/').removesuffix('\tHEAD')
  raise WorktreeError(
    f'Remote {remote!r} does not advertise a default branch. '
    'Set its HEAD to an existing branch, or create a checkout with '
    'Git/Jujutsu and open it with /folder.'
  )


def _create_git_worktree(source: pathlib.Path, name: str) -> Worktree:
  remote = _default_remote(source)
  warning = None
  # Fetch into a request-owned ref; FETCH_HEAD and tracking refs are shared.
  reference = f'refs/mirror/worktrees/{uuid.uuid4().hex}'
  try:
    if remote is None:
      revision = _git(source, ('rev-parse', '--verify', 'HEAD^{commit}'))
    else:
      try:
        _git(
          source,
          (
            'fetch',
            '--no-write-fetch-head',
            '--no-tags',
            '--refmap=',
            '--',
            remote,
            f'HEAD:{reference}',
          ),
          timeout=10,
        )
        revision = _git(source, ('rev-parse', reference))
      except (WorktreeError, subprocess.TimeoutExpired):
        try:
          revision = _git(
            source,
            ('rev-parse', '--verify', f'refs/remotes/{remote}/HEAD^{{commit}}'),
          )
          base = f'cached {remote}/HEAD'
        except WorktreeError:
          revision = _git(source, ('rev-parse', '--verify', 'HEAD^{commit}'))
          base = 'local HEAD (no cached default branch)'
        warning = (
          f'Could not refresh {remote!r}; created from {base} '
          f'at {revision[:12]}. It may be stale.'
        )
    common = pathlib.Path(
      _git(
        source,
        ('rev-parse', '--path-format=absolute', '--git-common-dir'),
      )
    ).resolve()
    branch = f'mirror/{name}'
    path = _managed_parent(common, common.parent.name or 'repository') / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(
      source,
      ('worktree', 'add', '-b', branch, str(path), revision),
    )
    resolved = path.resolve()
    return Worktree(
      resolved,
      branch,
      WorktreeIdentity('git', str(resolved)),
      warning=warning,
    )
  finally:
    _git(source, ('update-ref', '-d', reference))


def local_revision(source: pathlib.Path) -> tuple[pathlib.Path, str]:
  """Resolve the local committed revision without snapshotting dirty files."""
  root = project.repository_root(source.resolve())
  if root is None:
    raise WorktreeError(f'no Git or Jujutsu repository contains {source}')
  if project.jj_workspace_root(source) is not None:
    revision = _jj(
      root,
      (
        'log',
        '--ignore-working-copy',
        '--no-graph',
        '-r',
        '@-',
        '-T',
        'commit_id',
      ),
      timeout=120,
    )
    store = pathlib.Path(
      _jj(root, ('git', 'root', '--ignore-working-copy'), timeout=120)
    )
  else:
    revision = _git(
      root, ('rev-parse', '--verify', 'HEAD^{commit}'), timeout=120
    )
    store = pathlib.Path(
      _git(
        root,
        ('rev-parse', '--path-format=absolute', '--git-common-dir'),
        timeout=120,
      )
    )
  if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision) is None:
    raise WorktreeError('Repository copy requires one committed revision')
  return store, revision


def _add_jj_workspace(
  source: pathlib.Path, name: str, path: pathlib.Path, revision: str
) -> None:
  """Add a workspace, recovering a stale source checkout once."""
  arguments = (
    'workspace',
    'add',
    '--name',
    name,
    '--revision',
    revision,
    str(path),
  )
  try:
    _jj(source, arguments)
  except WorktreeError as error:
    if _JJ_STALE_WORKING_COPY not in str(error).lower():
      raise
    _jj(source, ('workspace', 'update-stale'))
    _jj(source, arguments)


def _jj_revision(source: pathlib.Path, revset: str) -> str:
  """Resolve a revset without snapshotting the working copy."""
  return _jj(
    source,
    (
      'log',
      '--ignore-working-copy',
      '--no-graph',
      '-r',
      revset,
      '-T',
      'commit_id',
    ),
  )


def _create_jj_workspace(
  source: pathlib.Path, root: pathlib.Path, name: str
) -> Worktree:
  store = pathlib.Path(_jj(source, ('git', 'root', '--ignore-working-copy')))
  remote = _default_remote(store)
  warning = None
  if remote is None:
    revision = _jj_revision(source, '@-')
  else:
    cached = _jj_revision(source, 'present(trunk())')
    base = 'the cached default branch'
    if not cached or cached == '0' * 40:
      cached = _jj_revision(source, '@-')
      base = 'the local committed revision (no cached default branch)'
    try:
      default_branch = _remote_default(store, remote)
      revset = (
        f'{json.dumps(default_branch, ensure_ascii=False)}@'
        f'{json.dumps(remote, ensure_ascii=False)}'
      )
      # Keep a pinned fallback even if fetch removes the remote bookmark.
      if remote_revision := _jj_revision(source, f'present({revset})'):
        cached = remote_revision
        base = 'the cached default branch'
      _jj(
        source,
        (
          'git',
          'fetch',
          '--ignore-working-copy',
          '--remote',
          f'exact:{json.dumps(remote, ensure_ascii=False)}',
          '--branch',
          f'exact:{json.dumps(default_branch, ensure_ascii=False)}',
        ),
        timeout=10,
      )
      revision = _jj_revision(source, revset)
    except (WorktreeError, subprocess.TimeoutExpired):
      revision = cached
      warning = (
        f'Could not refresh {remote!r}; created from {base} '
        f'at {revision[:12]}. It may be stale.'
      )
  # Resolve before workspace add: jj can leave a directory on revision errors.
  if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision) is None:
    raise WorktreeError(
      'Default branch must resolve to one commit; retry /worktree'
    )
  try:
    repository = project.jj_repository(root)
  except OSError as error:
    raise WorktreeError(
      f'could not read Jujutsu repository link: {error}'
    ) from error
  if repository is None:
    raise WorktreeError(f'no Jujutsu workspace contains {root}')
  parent = _managed_parent(
    repository, repository.parent.parent.name or 'repository'
  )
  path = parent / name
  path.parent.mkdir(parents=True, exist_ok=True)
  _add_jj_workspace(source, name, path, revision)
  return Worktree(
    path.resolve(), name, WorktreeIdentity('jj', name), warning=warning
  )


def fork_destination(cwd: pathlib.Path, name: str | None) -> pathlib.Path:
  """Reserve a managed parent directory, without creating or fetching a checkout."""
  root = project.repository_root(cwd.resolve())
  if root is None:
    raise WorktreeError('forking requires a Git or Jujutsu repository')
  repository = project.jj_repository(root)
  if repository is None:
    repository = pathlib.Path(
      _git(root, ('rev-parse', '--path-format=absolute', '--git-common-dir'))
    ).resolve()
  path = _managed_parent(repository, root.name) / _worktree_name(name)
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def create_worktree(
  cwd: str | pathlib.Path, name: str | None = None
) -> Worktree:
  """Create an isolated checkout, refreshing its local base when possible."""
  source = pathlib.Path(cwd).resolve()
  worktree_name = _worktree_name(name)
  if (jj_root := project.jj_workspace_root(source)) is not None:
    return _create_jj_workspace(source, jj_root, worktree_name)
  return _create_git_worktree(source, worktree_name)
