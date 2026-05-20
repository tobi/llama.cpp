# llama.cpp / examples / model-manager

Native C++ orchestrator that watches a directory of **Modelfiles** and drives
a running `./server` (load, unload, idle-out) over HTTP.

## Building

```bash
LLAMA_BUILD_SERVER=1 make
# or with CMake:
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build -j
```

This produces `./model-manager` alongside `./server`.

## Modelfile

A Modelfile is a small line-based file declaring a model and its runtime
parameters. Tokens are case-insensitive at the start of a line; `#` starts a
comment; values that span lines may be wrapped in `"""..."""`.

```text
NAME              llama2-7b-chat
FROM              models/7B/ggml-model.bin

SYSTEM """You are a precise assistant."""

TAG               default
TAG               chat

PARAMETER         n_ctx          2048
PARAMETER         n_gpu_layers   32
PARAMETER         temp           0.8
PARAMETER         top_p          0.95
PARAMETER         repeat_penalty 1.10

# Hardware-aware overrides. The single tier whose [low,high) GiB VRAM range
# covers the detected GPU wins; tiers are unioned by key so you only have to
# restate the value you want to specialise.
HARDWARE          vram_0_8     n_ctx        1024
HARDWARE          vram_0_8     n_gpu_layers 0
HARDWARE          vram_8_16    n_ctx        2048
HARDWARE          vram_8_16    n_gpu_layers 20
HARDWARE          vram_16_1000 n_gpu_layers 99
```

Supported keywords:

| Keyword             | Form                              | Effect                                  |
|---------------------|-----------------------------------|-----------------------------------------|
| `NAME`              | `NAME <alias>`                    | becomes `model_alias`                   |
| `FROM`              | `FROM <gguf-path>`                | required, becomes `params.model`        |
| `TEMPLATE`          | `TEMPLATE <line>` or `"""..."""`  | informational, surfaced in `/slots`     |
| `SYSTEM`            | `SYSTEM <line>` or `"""..."""`    | default system prompt                   |
| `PARAMETER`         | `PARAMETER <key> <value>`         | one of the keys below                   |
| `TAG`               | `TAG <tag>` (repeatable)          | used by autoload matching               |
| `SPECULATIVE_MODEL` | `SPECULATIVE_MODEL <path>`        | draft model (informational here)        |
| `SPECULATIVE_MAX`   | `SPECULATIVE_MAX <int>`           | max draft tokens                        |
| `SPECULATIVE_MIN`   | `SPECULATIVE_MIN <int>`           | min draft tokens                        |
| `HARDWARE`          | `HARDWARE vram_<a>_<b> <k> <v>`   | tier-specific PARAMETER override        |

`PARAMETER` keys map directly onto `gpt_params`. The common ones are
`n_ctx`, `n_batch`, `n_gpu_layers`, `main_gpu`, `tensor_split`, `temp`,
`top_p`, `top_k`, `tfs_z`, `typical_p`, `repeat_penalty`, `repeat_last_n`,
`frequency_penalty`, `presence_penalty`, `mirostat[_tau|_eta]`, `seed`,
`use_mmap`, `use_mlock`, `memory_f16`, `embedding`, `lora`, `lora_base`.

## Using a Modelfile directly

Every binary that uses `gpt_params` (main, server, perplexity, embedding,
…) accepts `--model-config <path>`:

```bash
./server --model-config models/example.modelfile --host 127.0.0.1 --port 8080
./main   --model-config models/example.modelfile -p "hello"
```

Hardware tiers are detected and applied automatically.

## Running the manager

```bash
./model-manager                       # uses ~/.config/llama/manager.conf
./model-manager /path/to/manager.conf # explicit
```

`manager.conf` is a simple `key value` file. See
`examples/model-manager/manager.conf.example` for the full template; the
defaults are equivalent to:

```text
server_url            http://127.0.0.1:8080
models_root           ~/.config/llama/models
idle_timeout_seconds  120
poll_interval_seconds 5
autoload              true
# autoload_tag        default
```

If `autoload_tag` is set (one per line, repeatable) only Modelfiles carrying
at least one of those tags are eligible for autoload. With no tags, any
Modelfile is eligible; ties are broken alphabetically.

The manager produces NDJSON on stdout — one event per line. Typical events:

```text
{"ts":...,"msg":"config-loaded","detail":"/.../manager.conf"}
{"ts":...,"msg":"hardware-detected","detail":{"vram_gb":24,"ram_gb":64,"gpu":"NVIDIA GeForce RTX 4090"}}
{"ts":...,"msg":"autoload-ok","detail":"llama2-7b-chat"}
{"ts":...,"msg":"idle-unload","detail":"llama2-7b-chat"}
{"ts":...,"msg":"server-down","detail":"err=2"}
```

## Server endpoints added for orchestration

The server gains four endpoints. They are safe to call repeatedly and the
server stays single-threaded (`CPPHTTPLIB_THREAD_POOL_COUNT == 1`), so loads
and completions can never race.

| Method | Path         | Body                                                                    | Returns |
|--------|--------------|-------------------------------------------------------------------------|---------|
| GET    | `/slots`     | —                                                                       | `{loaded, alias, model, source, n_ctx, n_gpu_layers, n_batch, last_activity_ms, idle_ms, tags[], speculative?}` |
| GET    | `/v1/models` | —                                                                       | OpenAI-shaped `data` list with the single loaded model |
| POST   | `/load`      | `{"model_config":"<path>"}` *or* `{"model":"<gguf>","alias":...,"n_ctx":...}` | `{ok:true, alias}` on success, 400 on parse error |
| POST   | `/unload`    | `{}`                                                                    | `{ok:true, was_loaded:bool}` |

`/completion`, `/tokenize`, and `/embedding` return `503` when no model is
loaded so cold-started servers fail loud rather than null-deref. They also
touch `last_activity_ms` on every successful call, which is what the manager
uses for idle-out.

## Quick smoke test (no GPU required)

```bash
# Cold-start the server (no -m)
./server --host 127.0.0.1 --port 8080 --model "" &

curl -s http://127.0.0.1:8080/slots                # {"loaded":false}
curl -s -X POST http://127.0.0.1:8080/load \
     -H 'Content-Type: application/json' \
     -d '{"model_config":"models/example.modelfile"}'
curl -s http://127.0.0.1:8080/slots                # populated
curl -s -X POST http://127.0.0.1:8080/unload       # {"ok":true,"was_loaded":true}
```

## What this is and isn't

It **is** a faithful native port of the orchestration spec — Modelfile
parsing, layered defaults via gpt_params, hardware-aware tier overrides,
autoload by tag, idle unload, NDJSON event log. It reuses every part of
the server that can be reused; the loader is shared via `examples/common`.

It **isn't** the YAML/JSON/INI formats from the spec — only Modelfiles
ship in this first cut (scope choice during build-out). It also doesn't
yet wire `TEMPLATE`/`SYSTEM`/`SPECULATIVE_*` into the inference loop —
those are parsed and surfaced via `/slots` but the actual chat-template
substitution and speculative decoding are bigger features that this
codebase doesn't yet have a runtime for.
