"""Choose and persist the active local web-search and browse provider."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from typing import override

from mirror import config
from mirror.cli import (
  actions,
  api_keys,
  commands,
  effects,
  extension,
  selection,
  state,
)
from mirror.cli.extensions.settings import setting_template


@dataclasses.dataclass(frozen=True)
class Choose(actions.ExtensionAction):
  provider: str = ''


def _route(provider: str) -> actions.RoutedExtensionAction:
  return actions.RoutedExtensionAction(
    WebProviderExtension.id(), Choose(provider)
  )


class WebProviderExtension(extension.Extension[None, Choose, None]):
  @classmethod
  @override
  def id(cls) -> str:
    return 'core/web_provider'

  @override
  def slash_commands(self) -> Sequence[commands.CommandDef]:
    return (
      commands.CommandDef(
        'web-provider',
        'Choose the local web-search and browse provider',
        lambda _context, args: _route(args.strip()),
        arguments=commands.OptionalArguments('[you|tavily|firecrawl]'),
        available_when_busy=True,
      ),
    )

  @override
  def reduce(
    self, global_state: state.State, action: actions.Action
  ) -> tuple[
    None, tuple[effects.CoreEffect | extension.PrivateEffect[Choose], ...]
  ]:
    del global_state  # Unused.
    if (
      isinstance(action, actions.RoutedExtensionAction)
      and action.extension_id == self.id()
      and isinstance(action.action, Choose)
    ):
      return None, (self.effect(action.action),)
    return None, ()

  @override
  async def handle_effect(
    self, context: extension.Context, effect: Choose
  ) -> None:
    if context.workspace.remote:
      context.dispatch(
        actions.ShowText(
          'Hosted sessions use the server’s web provider. This setting is local only.',
          style='ui.warning',
        )
      )
      return
    if effect.provider not in ('', 'you', 'tavily', 'firecrawl'):
      context.dispatch(
        actions.ShowText(
          'Usage: /web-provider [you|tavily|firecrawl]', style='ui.warning'
        )
      )
      return
    settings = setting_template.user_config(context)
    try:
      if not effect.provider:
        current = config.web_provider(settings.read()) or 'you'
        context.dispatch(
          actions.OpenSelector(
            selection.Selector(
              'web-provider',
              'Web provider',
              (
                selection.Choice(
                  'you',
                  'You.com',
                  f'you · {api_keys.environment_key_hint("YOU_API_KEY")}',
                ),
                selection.Choice(
                  'tavily',
                  'Tavily',
                  f'tavily · {api_keys.environment_key_hint("TAVILY_API_KEY")}',
                ),
                selection.Choice(
                  'firecrawl',
                  'Firecrawl',
                  f'firecrawl · {api_keys.environment_key_hint("FIRECRAWL_API_KEY")}',
                ),
              ),
              _route,
              current=current,
              extension_id=self.id(),
            )
          )
        )
        return
      key = f'{effect.provider.upper()}_API_KEY'
      if not os.environ.get(key, '').strip():
        context.dispatch(
          actions.ShowText(
            f'Set ${key} in your launch environment to use {effect.provider}. '
            'Web provider unchanged.',
            style='ui.warning',
          )
        )
        return
      settings.update(
        lambda values: values.update(web_provider=effect.provider)
      )
    except (OSError, ValueError):
      context.dispatch(
        actions.ShowText(
          'Could not read or save web_provider in config.json. Check the file and permissions.',
          style='ui.error',
        )
      )
      return
    context.dispatch(actions.SetWebProvider(effect.provider))
    context.dispatch(
      actions.ShowText(
        f'Web provider: {effect.provider}. Saved; applies to subsequent tool calls.',
        style='ui.info',
      )
    )
