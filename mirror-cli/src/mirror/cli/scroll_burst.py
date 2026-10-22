"""Application-owned redraw and full-GC scheduling for accepted wheel input."""

from __future__ import annotations

import asyncio
import gc
import sys
import threading

from prompt_toolkit.application import application

# Thresholds are process-wide: overlapping roots must not save each other's
# temporary policy. This lease only coordinates Mirror's scroll controllers.
_GC_LEASE = threading.Lock()
_DEFERRED_GENERATION_2 = 2**31 - 1


class ScrollBurst:
  """Temporarily relax one root's redraw throttle without forcing a draw.

  The owner calls `close` at view/terminal handoff and in its run's `finally`.
  Timers are cooperative, not real-time limits. Full collections still block,
  including when a new gesture coincides with the idle collection.
  """

  def __init__(self, app: application.Application[str | None]) -> None:
    self._app = app
    self._interval: float | None = None
    self._redraw_idle: asyncio.TimerHandle | None = None
    self._thresholds: tuple[int, int, int] | None = None
    self._gc_idle: asyncio.TimerHandle | None = None
    self._gc_deadline: asyncio.TimerHandle | None = None

  def touch(self) -> None:
    """Extend a burst before processing wheel input accepted by the active view."""
    if not self._app.is_running or self._app.is_done:
      return
    loop = asyncio.get_running_loop()
    if self._redraw_idle is None:
      self._interval = self._app.min_redraw_interval
    else:
      self._redraw_idle.cancel()
    self._app.min_redraw_interval = 0
    self._redraw_idle = loop.call_later(0.1, self._restore_redraw)
    self._defer_full_collection(loop)

  def _defer_full_collection(self, loop: asyncio.AbstractEventLoop) -> None:
    # The evidence covers CPython 3.12. In particular, newer collectors can
    # ignore threshold2 entirely; don't silently apply this policy to them.
    if sys.implementation.name != 'cpython' or sys.version_info[:2] != (3, 12):
      return
    thresholds = gc.get_threshold()
    if not gc.isenabled() or thresholds[0] == 0:
      return
    if self._thresholds is None:
      if thresholds[2] == _DEFERRED_GENERATION_2:
        return
      if not _GC_LEASE.acquire(blocking=False):
        return
      self._thresholds = gc.get_threshold()
      gc.set_threshold(*self._thresholds[:2], _DEFERRED_GENERATION_2)
      self._gc_deadline = loop.call_later(4, self._finish_gc)
    if self._gc_idle is not None:
      self._gc_idle.cancel()
    self._gc_idle = loop.call_later(0.25, self._finish_gc)

  def _restore_redraw(self) -> None:
    if self._redraw_idle is not None:
      self._redraw_idle.cancel()
      self._redraw_idle = None
      if self._app.min_redraw_interval == 0:
        self._app.min_redraw_interval = self._interval

  def _finish_gc(self) -> None:
    """Release our threshold override before a synchronous full collection."""
    thresholds = self._thresholds
    if thresholds is None:
      return
    self._thresholds = None
    if self._gc_idle is not None:
      self._gc_idle.cancel()
      self._gc_idle = None
    if self._gc_deadline is not None:
      self._gc_deadline.cancel()
      self._gc_deadline = None
    try:
      current = gc.get_threshold()
      # Preserve changes made by another policy while we owned the lease.
      # Direct GC users do not participate in it; this is only best effort.
      if current[2] == _DEFERRED_GENERATION_2:
        gc.set_threshold(*current[:2], thresholds[2])
        # An idle wheel gap is not itself a reason to traverse the whole heap.
        if (
          gc.isenabled()
          and current[0] != 0
          and gc.get_count()[2] > thresholds[2]
        ):
          gc.collect(2)
    finally:
      _GC_LEASE.release()

  def close(self) -> None:
    """Cancel timers and restore settings before relinquishing the application."""
    self._restore_redraw()
    self._finish_gc()
