"""Open a public GitHub issue form; never collect or submit session data."""

from __future__ import annotations

import dataclasses
import webbrowser
from collections.abc import Awaitable, Callable
from typing import override

from mirror.cli import actions, commands, extension, selection, state, terminal

_FORM_URL = 'https://github.com/reflection-oss/mirror/issues/new'
_INSTRUCTIONS = (
  'PUBLIC destination: https://github.com/reflection-oss/mirror\n\n'
  'Title and body start blank. Write your feedback on GitHub, review both, '
  'then click Create issue only if you want to publish them publicly.\n'
  'Mirror attaches nothing: no conversation, logs, files, or account details. '
  'Do not include secrets or private information. Report security issues '
  'privately to developers@reflection.ai instead.\n\n'
  'Sign in to GitHub in your browser if asked. Close the form to cancel. '
  'If submission fails, check existing issues before trying again.'
)


@dataclasses.dataclass(frozen=True)
class OpenForm(actions.ExtensionAction):
  """The user chose to continue to GitHub, not to publish an issue."""


def _confirm(_: commands.CommandContext, args: str) -> actions.Action:
  del args
  return actions.OpenSelector(
    selection.Selector(
      'feedback',
      'Public feedback',
      (
        selection.Choice('cancel', 'Cancel'),
        selection.Choice('open', 'Open GitHub form'),
      ),
      lambda choice: (
        actions.RoutedExtensionAction('feedback', OpenForm())
        if choice == 'open'
        else actions.ShowText('Feedback cancelled; nothing submitted.')
      ),
      selected='cancel',
      cancel_action=actions.ShowText('Feedback cancelled; nothing submitted.'),
      header=_INSTRUCTIONS,
    )
  )


class Feedback(extension.Extension[None, OpenForm, None]):
  """Hand authorship, authentication, final review, and submission to GitHub."""

  def __init__(
    self, open_url: Callable[[str], Awaitable[None]] = terminal.open_url
  ) -> None:
    self._open_url = open_url

  @classmethod
  @override
  def id(cls) -> str:
    return 'feedback'

  @override
  def slash_commands(self) -> tuple[commands.CommandDef, ...]:
    return (
      commands.CommandDef(
        'feedback',
        'Write feedback on public GitHub (review before posting)',
        _confirm,
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[None, tuple[extension.PrivateEffect[OpenForm], ...]]:
    del global_state
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, OpenForm)
    ):
      return None, (self.effect(action.action),)
    return None, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: OpenForm
  ) -> None:
    del effect
    context.dispatch(
      actions.ShowText(
        f'{_INSTRUCTIONS}\n\nIf the browser or network is unavailable, '
        f'open this address when ready:\n{_FORM_URL}\n'
        'Mirror cannot tell whether you submit an issue in the browser.'
      )
    )
    try:
      await self._open_url(_FORM_URL)
    except (OSError, webbrowser.Error):
      context.dispatch(
        actions.ShowText(
          f'Could not open a browser. Open {_FORM_URL} manually. '
          'Mirror has not submitted anything.',
          style='ui.warning',
        )
      )
