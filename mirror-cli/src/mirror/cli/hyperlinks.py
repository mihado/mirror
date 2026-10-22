"""Preserve native OSC 8 links through prompt-toolkit's ANSI and screen layers."""

from __future__ import annotations

import functools
import hashlib
import re
from typing import override

from prompt_toolkit import formatted_text, utils
from prompt_toolkit.formatted_text import ansi
from prompt_toolkit.output import base, vt100
from rich import color
from rich import style as rich_style

from mirror.cli import rendering

_OSC_LINK = re.compile(
  r'\x1b\]8;[^\x00-\x1f\x7f-\x9f;]*;([^\x00-\x1f\x7f-\x9f]*)(?:\x07|\x1b\\)'
)
_CLOSE = '\x1b]8;;\x1b\\'


@functools.lru_cache(maxsize=1024)
def _rich_style(sample: str) -> str:
  """Translate each visual style once, without retaining action metadata."""
  return formatted_text.to_formatted_text(ANSI(sample))[0][0]


def from_rich(
  value: str, appearance: rich_style.Style | None, system: color.ColorSystem
) -> formatted_text.StyleAndTextTuples:
  """Keep Rich spans intact instead of ANSI-decoding every character.

  Links still need cell-local escapes, and embedded ANSI still needs parsing.
  Use the existing decoder for both so terminal sanitization stays identical.
  """
  if (appearance and appearance.link) or '\x1b' in value or '\x9b' in value:
    if appearance:
      value = appearance.render(value, color_system=system)
    return formatted_text.to_formatted_text(ANSI(value))
  sample = appearance.render('x', color_system=system) if appearance else 'x'
  return [(_rich_style(sample), rendering.strip_ansi(value))]


class Output(vt100.Vt100_Output):
  """Close each cell's hyperlink before the renderer skips or moves the cursor."""

  _link_open = False

  @override
  def write_raw(self, data: str) -> None:
    if match := _OSC_LINK.fullmatch(data):
      self._link_open = bool(match[1])
    super().write_raw(data)

  @override
  def write(self, data: str) -> None:
    super().write(data)
    if self._link_open:
      self.write_raw(_CLOSE)


def native_output(output: base.Output) -> base.Output:
  """Retain the default output's capabilities, adding OSC 8 support on VT100."""
  if not isinstance(output, vt100.Vt100_Output):
    return output
  return Output(
    output.stdout,
    output.get_size,
    term=output.term,
    default_color_depth=output.default_color_depth,
    enable_bell=output.enable_bell,
    enable_cpr=output.enable_cpr,
  )


class ANSI(ansi.ANSI):
  """Carry sanitized links as zero-width escapes, never mouse handlers."""

  def __init__(self, value: str) -> None:
    super().__init__(
      _OSC_LINK.sub(
        lambda match: '\x01' + match[0] + '\x02', rendering.sanitize_ansi(value)
      )
    )
    self._fragments: formatted_text.StyleAndTextTuples = []
    link, link_style = '', ''
    for style, text, *_ in super().__pt_formatted_text__():
      if style == '[ZeroWidthEscape]':
        match = _OSC_LINK.fullmatch(text)
        url = match[1] if match else ''
        link, link_style = '', ''
        if url:
          identity = hashlib.sha256(url.encode()).hexdigest()
          link = f'\x1b]8;id={identity};{url}\x1b\\'
          # The screen diff must also repaint a label whose destination changed.
          link_style = f' class:hyperlink-{identity}'
        continue
      # Each painted cell is independent; surrounding cells may be skipped.
      if link and utils.get_cwidth(text) > 0:
        self._fragments.append(('[ZeroWidthEscape]', link))
      self._fragments.append((style + link_style, text))

  @override
  def __pt_formatted_text__(self) -> formatted_text.StyleAndTextTuples:
    return self._fragments
