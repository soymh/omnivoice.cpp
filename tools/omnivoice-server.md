# omnivoice-server

Full-featured HTTP server for OmniVoice TTS with OpenAI-compatible endpoints,
per-request parameter overrides, and per-request voice cloning via base64-encoded
reference audio.

## Endpoints

| Method | Path                 | Description                                      |
|--------|----------------------|--------------------------------------------------|
| POST   | `/v1/audio/speech`   | OpenAI-compatible TTS synthesis                  |
| POST   | `/synthesize`        | Extended synthesis (same handler as above)        |
| GET    | `/v1/models`         | List the loaded model                            |
| GET    | `/v1/voices`         | List named speakers (empty for OmniVoice)        |
| GET    | `/health`            | Liveness probe `{"status":"ok"}`                 |
| GET    | `/props`             | Server configuration and capabilites             |

## Usage

```
omnivoice-server --model <gguf> --codec <gguf> [options]
```

### Required

| Flag | Description |
|------|-------------|
| `--model <gguf>` | LLM GGUF (F32 / BF16 / Q8_0) |
| `--codec <gguf>` | Codec GGUF (omnivoice-tokenizer-*.gguf) |

### Server

| Flag | Default | Description |
|------|---------|-------------|
| `--host <ip>` | `127.0.0.1` | Listen address |
| `--port <n>` | `8080` | Listen port |
| `--parallel <n>` | `1` | Max concurrent requests |
| `--timeout-read <sec>` | `60` | HTTP read timeout |
| `--timeout-write <sec>` | `120` | HTTP write timeout |

### Hardware

| Flag | Description |
|------|-------------|
| `--device <name>` | Force a specific backend device (CUDA0, Vulkan0, Metal, CPU) |
| `--threads <n>` | OpenMP thread count (default: auto) |
| `--numa <mode>` | NUMA policy: `distribute`, `isolate`, `numactl` |
| `--no-fa` | Disable flash attention |
| `--flash-attn <on\|off>` | Flash attention override |
| `--clamp-fp16` | Clamp hidden states to FP16 range |

### Synthesis Defaults (overridable per-request)

| Flag | Default | Description |
|------|---------|-------------|
| `--format <fmt>` | `wav16` | WAV format: `wav16`, `wav24`, `wav32` |
| `--lang <str>` | (none) | Language label |
| `--instruct <str>` | (none) | Style instruction |
| `--duration <sec>` | `0` (auto) | Force output duration |
| `--no-denoise` | (enabled) | Omit denoising prefix |
| `--no-preprocess-prompt` | (enabled) | Skip reference pre-processing |
| `--chunk-duration <sec>` | `15.0` | Long-form chunk duration |
| `--chunk-threshold <sec>` | `30.0` | Activate chunking above this |
| `--seed <int>` | `-1` (random) | Sampling seed |

### Server-wide Reference Audio

| Flag | Description |
|------|-------------|
| `--ref-wav <path>` | Reference WAV for voice cloning |
| `--ref-rvq <path>` | Pre-encoded reference codes (replaces --ref-wav) |
| `--ref-text <path>` | Transcript file for the reference |

### Debug

| Flag | Description |
|------|-------------|
| `--dump <dir>` | Dump intermediate tensors to `<dir>` |

## Per-Request Overrides

All synthesis defaults can be overridden per-request by including extra fields
in the JSON body of `POST /v1/audio/speech` or `POST /synthesize`:

```json
{
  "input": "Hello world",
  "lang": "en",
  "instruct": "cheerful",
  "duration": 5.0,
  "seed": 42,
  "no_denoise": false,
  "chunk_duration": 10.0,
  "chunk_threshold": 20.0,
  "format": "wav24",
  "stream": false
}
```

### Per-Request Voice Cloning

Include a base64-encoded WAV file and optional transcript:

```json
{
  "input": "Hello world",
  "ref_wav_base64": "UklGRiR...AAA=",
  "ref_text": "Original utterance transcript"
}
```

When `ref_wav_base64` is absent, the server-wide reference (loaded via
`--ref-wav` / `--ref-rvq`) is used if available.
