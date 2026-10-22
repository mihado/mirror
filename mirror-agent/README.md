# Mirror Agent

Mirror Agent runs a coding agent over a local HTTP API. Send prompts, let it
read and edit files or run commands, and collect the conversation as JSONL.
Use it for automation, evaluations, and embedding in other applications.

## Install

The PyPI package `reflection-mirror` installs the `mirror` CLI. This page covers
the standalone native executable and its HTTP API.

From the [full source checkout](../README.md#run-from-source), install Rust,
a C compiler, `protoc`, and ripgrep (`rg`), then run at the repository root:

```bash
cargo install --locked --path mirror-agent --bin mirror-agent
mirror-agent --version
```

Ensure Cargo's binary directory (usually `~/.cargo/bin`) is on `PATH`.
Keep the accompanying license and attribution files when redistributing binaries.

### Requirements

- **Linux x86-64 or ARM64.**
- **Model access:** a supported provider or an OpenAI-compatible endpoint.
- **A writable workspace**, plus an output directory for file traces.

Source-built binaries use ripgrep from `PATH`; local shell tools require
`/bin/bash`. Python and uv are not needed to run the executable. Install any
compilers, test runners, or other tools your project needs separately.

## First conversation

**Run inside an isolated environment. The HTTP API has no authentication, and
all tools execute without approval prompts.** Keep the default loopback bind;
do not expose the server publicly. `--workspace` chooses a working directory,
not a sandbox. Do not give the environment unrelated credentials or host access.

Start with an OpenAI-compatible Chat Completions endpoint:

```bash
# Only needed if your model endpoint requires authentication.
export OPENAI_API_KEY="your-model-api-key"

mirror-agent serve \
  --api openai-chat-completions \
  --base-url https://model-endpoint.example/v1 \
  --model your-model-name \
  --workspace /path/to/project \
  --output ./mirror.jsonl
```

The key authenticates requests to the model, **not** requests to Mirror Agent.
Use a dedicated, narrowly scoped credential if one is required.

From another terminal in the same environment:

```bash
curl --fail-with-body -i http://127.0.0.1:4097/health

curl --fail-with-body -i http://127.0.0.1:4097/message \
  -H 'Content-Type: text/plain' \
  --data-binary 'Explain how this project is organized.'

cat mirror.jsonl
```

`/health` returns `204`. `/message` accepts a plain-text UTF-8 prompt, waits for
the turn to finish, and returns `204` with no body. Read replies and tool results
from the output file. Successive prompts share one conversation; concurrent
prompts return `409`. Prompt bodies are limited to 8 MiB; requests to `/message`
with an `Origin` header are rejected.

## Configure

Run `mirror-agent serve --help` for all options.
Pass MCP connections as `--mcp-connections '<JSON array>'`; see [MCP](docs/configuration.md#mcp).

| Setting                                | Option                                                                              |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| Provider protocol, endpoint, and model | `--api`, `--base-url`, `--model`                                                    |
| Working directory                      | `--workspace`                                                                       |
| Available tools                        | `--tools bash` (default); also `read,write,edit,grep,find,ls,web_search,web_browse` |
| Native builtin declarations            | `--use-native-bash`, `--use-native-web` (Chat Completions only)                     |
| Set the system prompt                  | `--system-prompt`                                                                   |
| Opt in to project instructions         | `--include-project-instructions`                                                    |
| Opt in to the skills catalog           | `--include-skills`                                                                  |
| Reasoning level or token budget        | `--reasoning-effort`                                                                |
| Sampling                               | `--temperature`, `--top-p`, `--seed`                                                |
| Context and output token limits        | `--context-window`, `--max-tokens`                                                  |
| Per-turn limits                        | `--max-iterations`, `--turn-timeout-seconds`, `--request-timeout-seconds`           |
| Provider-specific request fields       | `--extra-body` (JSON object)                                                        |
| Personal memory                        | `--memory-dir` (see [Memory](docs/memory.md))                                       |

Web search and browsing default to You.com (`YOU_API_KEY`): `rerank_v5` snippets
with a 200-character target, and full Markdown through Contents with a 60-second
crawl timeout. The CLI's `/web-provider` extension persists selection in
`~/.mirror/config.json` and sends selection through AgentRPC without restarting.
Local tools construct adapters on demand; selection is separate from inference settings.
Native headless runs use `--web-provider you|tavily|firecrawl`; they do not read
the CLI config. Alternatives:

- **Tavily** (`TAVILY_API_KEY`): basic Search without generated answers or automatic
  parameter selection; basic Extract in Markdown without query/chunk filtering.
  Extract does not supply a title or promise a redirect destination.
- **Firecrawl** (`FIRECRAWL_API_KEY`): v2 Search with web snippets and highlights
  disabled; v2 Scrape with full Markdown, no LLM cleaning, TLS verification, and
  no writes to Firecrawl's page cache. Titles and final URLs use scrape metadata
  when available. Provider-side caching and retrieval may still differ.

Selection applies to both tools, independently of the model provider. Mixed
providers and custom API endpoints are not supported. Missing keys never block
agent startup or recovery; unavailable web calls fail without fallback.
The local CLI omits web tools from model context when the selected provider's
key is absent or blank, even if another provider has a key. Live switches update
visibility on subsequent model requests.
Keys use the existing environment-based
API compatibility path, stay out of command arguments, and are removed from bash
and ripgrep children. Changes apply to subsequent tool batches; running calls keep
their selection. Page caches are partitioned by provider within the conversation,
including redirect aliases. Retrieval content and quality can differ.

With `--tools web_search,web_browse --use-native-web`, tools are advertised and
called as `search` and `browse`; only their declarations change. Native flags
can be combined. `--remote` supports host-side search, but still rejects browse.
Hosted sessions retain the server's You.com adapter; local selection does not affect them.

By default, model requests have no system message. Project instructions
(`AGENTS.md` / `CLAUDE.md`, including global instructions) and skills are disabled
unless explicitly enabled with the flags above. `--system-prompt` adds instructions before any
opted-in context. The server never adds an environment section. Reasoning and
sampling support vary by provider.

## Output and lifecycle

`--in-memory-output` keeps traces in memory instead of writing files:

- `GET /trace`: committed conversation records as NDJSON.
- `GET /trace/model-calls`: model-call diagnostics as NDJSON.
- Add `--output PATH` to also write files.
- Collect before shutdown; traces are lost on exit or crash.
- Reads are unauthenticated. This avoids filesystem discovery, not deliberate
  access by the agent.

Without `--in-memory-output`, file output remains the default.
**Starting the server truncates existing output files. There is no resume
support.** Choose a new `--output` path for each run you want to preserve.
Relative paths resolve from the launch directory, not `--workspace`; missing
parent directories are created.

| File (with `--output mirror.jsonl`) | Contents |
| --- | --- |
| `mirror.jsonl` | Committed `AgentMessage` records, one protobuf-JSON object per line |
| `mirror.model-calls.jsonl` | Model-facing messages, tools, settings, usage, and errors |

Conversation records are flushed as they are committed, not streamed token by
token. Model-call traces include `context_reset` records that re-emit the active
context; interrupted calls may have no assistant response. Model-call trace
write failures warn on stderr without stopping inference.

These files can contain source code, prompts, and tool output. Treat them as
sensitive. Model requests send context to the configured provider.

Stop with Ctrl-C or SIGTERM to cancel active work and shut down. Stop the
server before rebuilding or replacing the executable, check its version, and
start it with a fresh output path. Keep a copy of the old executable and
outputs if you need to reproduce a run.

## Documentation

See the [documentation index](docs/README.md) for architecture, configuration,
and HTTP/AgentRPC interface guides.

## License

Mirror Agent is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for
attribution. Third-party components retain their respective licenses.
