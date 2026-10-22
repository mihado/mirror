# Changelog

CLI releases use `reflection-mirror-vX.Y.Z`; agent releases use `mirror-agent-vX.Y.Z`.

## [reflection-mirror-v0.1.1]

### Added

- `/feedback` for sharing feedback and reporting issues.
- `/fork [name]` to copy an idle local conversation and its working files into a
  new workspace.
- In-app inspection of sub-agent conversations.
- `/web-provider` to choose You.com, Tavily, or Firecrawl for search and browsing,
  with the choice saved across launches. You.com is the default.
- Display of recorded session usage.
- Support for loading extensions directly from local source directories.
- Batch extension approvals and visibility into failed or unapproved extensions
  in `/extensions`.
- Tools can require confirmation even in automatic approval mode.
- Support for additional Gemini models.
- Optional HTTP endpoints for in-memory agent traces.

### Changed

- Smoother transcript scrolling and streaming, with fewer redraws and fewer
  stalls in long conversations.
- Mouse-wheel events now scroll one row at a time.
- When scrolled up, Escape returns to the latest output before interrupting
  active work.
- Automatic terminal background detection and improved light-theme syntax colors.
- Improved highlighting for tool actions, quoted strings, and Bash commands.
- Terminal tabs show the active workspace name.
- Clearer workspace choices in `Ctrl+N`.
- More compact elapsed-time labels, single-row running summaries, and clearer
  tool progress groups.
- More consistent web-search and browsing output.
- Concurrency-safe tool calls can run in parallel.
- Persistent WebSocket connections for supported OpenAI Responses endpoints.
- macOS wheels now support macOS 14 and later.
- Expanded usage analytics and documented collection and opt-out controls.

### Fixed

- Clipboard copying in local Apple Terminal sessions.
- Activity-layout warnings triggered by scrolling.
- Name broken configuration files in error messages.
- Include failure reasons in `mirror exec` errors.
- Initialize HTTP MCP clients for each run.
- Avoid repeated launch screens and gaps around the scrolling status area.
- Limit repeated malformed tool turns using the consecutive-error budget.
- Coerce tool arguments to their declared types.
- Correct Anthropic usage accounting, prompt caching, and thinking-history replay.

### Security

- Updated TLS trust stores and certifi certificate data.

### Contributors

- @akseljoonas
- @ChetanTekur
- @drubinstein
- @germtb
- @jnations-reflection
- @m1k3s0 — reference implementation for the Apple Terminal clipboard fix.
- @nschank
- @sndrws
