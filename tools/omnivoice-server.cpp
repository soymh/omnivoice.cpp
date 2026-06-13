// omnivoice-server.cpp: full-featured HTTP server for OmniVoice TTS.
//
// Serves OpenAI-compatible POST /v1/audio/speech and the extended
// POST /synthesize endpoint with per-request parameter overrides,
// per-request voice cloning via base64-encoded reference WAV, and
// server-wide reference audio from CLI flags.
//
// The shared server core (route handlers, OAI JSON parsing, audio
// encoding, streaming) lives in src/tts-server.h ; this file only
// wires the omnivoice ABI into the tts_backend adapter.

#include "tts-server.h"

#include "omnivoice.h"
#include "version.h"

#include "rvq-file.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
            "omnivoice.cpp %s\n\n"
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          LLM GGUF (F32 / BF16 / Q8_0)\n"
            "  --codec <gguf>          Codec GGUF (omnivoice-tokenizer-*.gguf)\n\n"
            "Server:\n"
            "  --host <ip>             Listen address (default: 127.0.0.1)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --parallel <n>          Max concurrent requests (default: 1)\n"
            "  --timeout-read <sec>    HTTP read timeout (default: 60)\n"
            "  --timeout-write <sec>   HTTP write timeout (default: 120)\n\n"
            "Hardware:\n"
            "  --device <name>         Force a specific backend device\n"
            "                          (CUDA0, Vulkan0, Metal, CPU, ...)\n"
            "  --threads <n>           OpenMP thread count (default: auto)\n"
            "  --numa <mode>           NUMA policy: distribute, isolate, numactl\n"
            "  --no-fa                 Disable flash attention\n"
            "  --flash-attn <on|off>   Flash attention override\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n\n"
            "Synthesis defaults (overridable per-request in JSON body):\n"
            "  --format <fmt>          Default WAV format: wav16, wav24, wav32 (default: wav16)\n"
            "  --lang <str>            Language label (default: none)\n"
            "  --instruct <str>        Style instruction (default: none)\n"
            "  --duration <sec>        Output duration in seconds (0 = auto)\n"
            "  --no-denoise            Omit the <|denoise|> prefix\n"
            "  --no-preprocess-prompt  Skip ref-wav silence trim / terminal punctuation\n"
            "  --chunk-duration <sec>  Long-form chunk duration (default: 15.0)\n"
            "  --chunk-threshold <sec> Activate chunking above this duration (default: 30.0)\n"
            "  --seed <int>            Sampling seed (-1 for random per-request)\n\n"
            "Server-wide reference audio:\n"
            "  --ref-wav <path>        Reference WAV for voice cloning\n"
            "  --ref-rvq <path>        Pre-encoded reference codes (replaces --ref-wav)\n"
            "  --ref-text <path>       Transcript file for the reference\n\n"
            "Debug:\n"
            "  --dump <dir>            Dump intermediate tensors to <dir>\n\n"
            "Endpoints:\n"
            "  POST /v1/audio/speech   OpenAI-compatible TTS\n"
            "  POST /synthesize        Extended synthesis with per-request overrides\n"
            "  GET  /v1/models         List loaded model\n"
            "  GET  /v1/voices         List named speakers (none for OmniVoice)\n"
            "  GET  /health            Liveness probe\n"
            "  GET  /props             Server configuration\n",
            OMNIVOICE_VERSION, prog);
}

// ---------------------------------------------------------------------------
// Base64 decode (RFC 4648). Skips whitespace, stops at '=' padding.
// ---------------------------------------------------------------------------
static const uint8_t b64_dec_tab[256] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255, 62,255,255,255, 63,
     52, 53, 54, 55, 56, 57, 58, 59, 60, 61,255,255,255,255,255,255,
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,255,255,255,255,255,
    255, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
     41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
};

static std::string base64_decode(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3 / 4);
    uint32_t buf = 0;
    int      bits = 0;
    for (char c : in) {
        if (c == '=') {
            break;
        }
        uint8_t d = b64_dec_tab[(uint8_t) c];
        if (d == 255) {
            continue;
        }
        buf = (buf << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((buf >> bits) & 0xFF));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Decode raw WAV bytes (from memory) into mono 24 kHz float samples.
// Caller must free the returned pointer. Returns NULL on failure.
// ---------------------------------------------------------------------------
static float * wav_from_bytes(const std::string & bytes, int * n_samples_out) {
    if (bytes.empty()) {
        *n_samples_out = 0;
        return NULL;
    }
    const uint8_t * data = (const uint8_t *) bytes.data();
    size_t          size = bytes.size();

    int     T  = 0, sr = 0;
    float * planar = audio_io_read_wav_buf(data, size, &T, &sr);
    if (!planar) {
        *n_samples_out = 0;
        return NULL;
    }

    if (sr != 24000) {
        float * resampled = audio_resample(planar, T, sr, 24000, 2, &T);
        free(planar);
        planar = resampled;
        sr = 24000;
        if (!planar) {
            *n_samples_out = 0;
            return NULL;
        }
    }

    float * mono = (float *) malloc((size_t) T * sizeof(float));
    if (!mono) {
        free(planar);
        *n_samples_out = 0;
        return NULL;
    }
    const float * left  = planar;
    const float * right = planar + (size_t) T;
    for (int i = 0; i < T; i++) {
        mono[i] = 0.5f * (left[i] + right[i]);
    }
    free(planar);
    *n_samples_out = T;
    return mono;
}

// ---------------------------------------------------------------------------
// Read a small text file (transcript) into a string. Trims trailing newlines.
// ---------------------------------------------------------------------------
static bool read_text_file(const char * path, std::string & out) {
    FILE * f = utf8_fopen(path, "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t) sz);
    if (sz > 0 && fread(&out[0], 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        return false;
    }
    fclose(f);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return true;
}

static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

// ---------------------------------------------------------------------------
// Parse --numa value into ggml_numa_strategy. Returns -1 on unknown.
// ---------------------------------------------------------------------------
static int parse_numa(const char * s) {
    if (!s) {
        return -1;
    }
    if (!strcmp(s, "distribute")) {
        return GGML_NUMA_STRATEGY_DISTRIBUTE;
    }
    if (!strcmp(s, "isolate")) {
        return GGML_NUMA_STRATEGY_ISOLATE;
    }
    if (!strcmp(s, "numactl")) {
        return GGML_NUMA_STRATEGY_NUMACTL;
    }
    return -1;
}

static const int RVQ_CODE_BITS = 11;  // 11 bits per code, matching omnivoice-codec

// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    utf8_init(&argc, &argv);
    try {
        // -- CLI defaults
        const char * model_path    = NULL;
        const char * codec_path    = NULL;
        server_config cfg;
        WavFormat    wav_fmt            = WAV_S16;
        const char * prompt_lang        = NULL;
        const char * prompt_instruct    = NULL;
        float        prompt_duration_sec = 0.0f;
        bool         prompt_denoise     = true;
        bool         preprocess_prompt  = true;
        float        chunk_duration_sec = 15.0f;
        float        chunk_threshold_sec = 30.0f;
        const char * ref_wav_path       = NULL;
        const char * ref_rvq_path       = NULL;
        const char * ref_text_path      = NULL;
        int          seed_arg           = -1;
        bool         use_fa             = true;
        bool         clamp_fp16         = false;
        const char * dump_dir           = NULL;
        int          threads_arg        = -1;
        const char * device_arg         = NULL;
        int          numa_arg           = -1;

        // -- Parse CLI
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "--codec") == 0 && i + 1 < argc) {
                codec_path = argv[++i];
            } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
                cfg.host = argv[++i];
            } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
                cfg.port = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--parallel") == 0 && i + 1 < argc) {
                cfg.n_parallel = atoi(argv[++i]);
                if (cfg.n_parallel < 1) {
                    cfg.n_parallel = 1;
                }
            } else if (strcmp(argv[i], "--timeout-read") == 0 && i + 1 < argc) {
                cfg.read_timeout_sec = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--timeout-write") == 0 && i + 1 < argc) {
                cfg.write_timeout_sec = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
                device_arg = argv[++i];
            } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
                threads_arg = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--numa") == 0 && i + 1 < argc) {
                numa_arg = parse_numa(argv[++i]);
                if (numa_arg < 0) {
                    fprintf(stderr, "[CLI] ERROR: unknown --numa mode: %s\n", argv[i]);
                    print_usage(argv[0]);
                    return 1;
                }
            } else if (strcmp(argv[i], "--no-fa") == 0) {
                use_fa = false;
            } else if (strcmp(argv[i], "--flash-attn") == 0 && i + 1 < argc) {
                const char * v = argv[++i];
                if (strcmp(v, "on") == 0 || strcmp(v, "auto") == 0) {
                    use_fa = true;
                } else if (strcmp(v, "off") == 0) {
                    use_fa = false;
                } else {
                    fprintf(stderr, "[CLI] ERROR: --flash-attn must be on|off|auto\n");
                    return 1;
                }
            } else if (strcmp(argv[i], "--clamp-fp16") == 0) {
                clamp_fp16 = true;
            } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
                if (!audio_parse_format(argv[++i], wav_fmt)) {
                    fprintf(stderr, "[CLI] ERROR: unknown format: %s\n", argv[i]);
                    print_usage(argv[0]);
                    return 1;
                }
            } else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) {
                prompt_lang = argv[++i];
            } else if (strcmp(argv[i], "--instruct") == 0 && i + 1 < argc) {
                prompt_instruct = argv[++i];
            } else if (strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
                prompt_duration_sec = (float) atof(argv[++i]);
            } else if (strcmp(argv[i], "--no-denoise") == 0) {
                prompt_denoise = false;
            } else if (strcmp(argv[i], "--no-preprocess-prompt") == 0) {
                preprocess_prompt = false;
            } else if (strcmp(argv[i], "--chunk-duration") == 0 && i + 1 < argc) {
                chunk_duration_sec = (float) atof(argv[++i]);
            } else if (strcmp(argv[i], "--chunk-threshold") == 0 && i + 1 < argc) {
                chunk_threshold_sec = (float) atof(argv[++i]);
            } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
                seed_arg = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--ref-wav") == 0 && i + 1 < argc) {
                ref_wav_path = argv[++i];
            } else if (strcmp(argv[i], "--ref-rvq") == 0 && i + 1 < argc) {
                ref_rvq_path = argv[++i];
            } else if (strcmp(argv[i], "--ref-text") == 0 && i + 1 < argc) {
                ref_text_path = argv[++i];
            } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
                dump_dir = argv[++i];
            } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
                print_usage(argv[0]);
                return 0;
            } else {
                fprintf(stderr, "[CLI] ERROR: unknown arg: %s\n", argv[i]);
                print_usage(argv[0]);
                return 1;
            }
        }

        if (!model_path || !codec_path) {
            print_usage(argv[0]);
            return 1;
        }

        // -- Validate reference audio flags
        if (ref_wav_path && ref_rvq_path) {
            fprintf(stderr, "[CLI] ERROR: --ref-wav and --ref-rvq are mutually exclusive\n");
            return 1;
        }
        if ((ref_wav_path || ref_rvq_path) && !ref_text_path) {
            fprintf(stderr, "[CLI] ERROR: --ref-wav / --ref-rvq requires --ref-text <path>\n");
            return 1;
        }

        // -- Apply env-var overrides for hardware config
        //    Device: backend_init() in backend.h checks GGML_BACKEND env var
        if (device_arg) {
            if (setenv("GGML_BACKEND", device_arg, 1) != 0) {
                fprintf(stderr, "[CLI] WARN: cannot set GGML_BACKEND\n");
            }
        }
        //    Threads: OpenMP / BLAS backend thread count
        if (threads_arg > 0) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%d", threads_arg);
            if (setenv("OMP_NUM_THREADS", buf, 1) != 0) {
                fprintf(stderr, "[CLI] WARN: cannot set OMP_NUM_THREADS\n");
            }
        }

        // -- NUMA init (must happen before any backend/context allocation)
        if (numa_arg >= 0) {
            ggml_numa_init((enum ggml_numa_strategy) numa_arg);
        }

        // -- Load model
        struct ov_init_params iparams;
        ov_init_default_params(&iparams);
        iparams.model_path = model_path;
        iparams.codec_path = codec_path;
        iparams.use_fa     = use_fa;
        iparams.clamp_fp16 = clamp_fp16;

        struct ov_context * ov = ov_init(&iparams);
        if (!ov) {
            fprintf(stderr, "[Server] FATAL: %s\n", ov_last_error());
            return 1;
        }

        // -- Load server-wide reference audio
        std::vector<float>   server_ref_audio;
        int                  server_ref_n_samples = 0;
        std::vector<int32_t> server_ref_tokens;
        int                  server_ref_T = 0;
        std::string          server_ref_text;

        if (ref_text_path) {
            if (!read_text_file(ref_text_path, server_ref_text)) {
                fprintf(stderr, "[Server] FATAL: cannot read --ref-text %s\n", ref_text_path);
                ov_free(ov);
                return 1;
            }
        }
        if (ref_wav_path) {
            fprintf(stderr, "[Server] Loading reference WAV: %s\n", ref_wav_path);
            int     n_samples = 0;
            float * raw       = audio_read_mono(ref_wav_path, 24000, &n_samples);
            if (!raw || n_samples <= 0) {
                fprintf(stderr, "[Server] FATAL: failed to load reference WAV: %s\n", ref_wav_path);
                free(raw);
                ov_free(ov);
                return 1;
            }
            server_ref_audio.assign(raw, raw + n_samples);
            server_ref_n_samples = n_samples;
            free(raw);
        }
        if (ref_rvq_path) {
            const int K = ov_num_codebooks(ov);
            if (!rvq_read_file(ref_rvq_path, K, RVQ_CODE_BITS, server_ref_tokens, &server_ref_T)) {
                fprintf(stderr, "[Server] FATAL: failed to load ref RVQ: %s\n", ref_rvq_path);
                ov_free(ov);
                return 1;
            }
            fprintf(stderr, "[Server] Loading reference RVQ: %s, K=%d T=%d\n", ref_rvq_path, K, server_ref_T);
        }

        // -- Build tts_backend adapter
        tts_backend be;
        be.model_id = basename_of(model_path);
        be.wav_fmt  = wav_fmt;

        be.synthesize = [ov, prompt_lang, prompt_instruct, prompt_duration_sec,
                         prompt_denoise, preprocess_prompt,
                         chunk_duration_sec, chunk_threshold_sec,
                         seed_arg, dump_dir,
                         &server_ref_audio, server_ref_n_samples,
                         &server_ref_tokens, server_ref_T,
                         &server_ref_text](
                            const tts_request & req,
                            const tts_sink &    sink,
                            std::string &       err) -> int {
            ov_tts_params p;
            ov_tts_default_params(&p);

            // Input text
            p.text = req.input.c_str();

            // Language: CLI default, overridden by per-request
            std::string lang;
            if (req.ov_lang.has_value()) {
                lang = req.ov_lang.value();
            } else if (prompt_lang) {
                lang = prompt_lang;
            }
            p.lang = lang.empty() ? "" : lang.c_str();

            // Instruct: per-request overrides CLI default, fallback to OAI instructions
            std::string instruct;
            if (req.ov_instruct.has_value()) {
                instruct = req.ov_instruct.value();
            } else if (prompt_instruct) {
                instruct = prompt_instruct;
            } else if (!req.instructions.empty()) {
                instruct = req.instructions;
            }
            p.instruct = instruct.c_str();

            // Duration override
            float duration_sec = prompt_duration_sec;
            if (req.ov_duration_sec.has_value()) {
                duration_sec = req.ov_duration_sec.value();
            }
            p.T_override = 0;
            if (duration_sec > 0.0f) {
                p.T_override = ov_duration_sec_to_tokens(ov, duration_sec);
            }

            // Chunk settings
            p.chunk_duration_sec  = req.ov_chunk_duration_sec.value_or(chunk_duration_sec);
            p.chunk_threshold_sec = req.ov_chunk_threshold_sec.value_or(chunk_threshold_sec);

            // Denoise
            p.denoise = req.ov_no_denoise.has_value() ? (!req.ov_no_denoise.value()) : prompt_denoise;

            // Preprocess prompt (server-wide only, sensible default)
            p.preprocess_prompt = preprocess_prompt;

            // Seed: per-request overrides CLI, CLI overrides random
            uint64_t seed;
            if (req.ov_seed.has_value()) {
                seed = (uint64_t) req.ov_seed.value();
            } else if (seed_arg >= 0) {
                seed = (uint64_t) seed_arg;
            } else {
                seed = (uint64_t) std::random_device{}();
            }
            p.mg_seed = seed;

            // Reference audio: per-request base64 WAV takes priority
            std::vector<float> local_ref_audio;
            std::string        local_ref_text;
            bool               has_local_ref = false;

            if (req.ov_ref_wav_base64.has_value()) {
                std::string wav_bytes = base64_decode(req.ov_ref_wav_base64.value());
                if (!wav_bytes.empty()) {
                    int ns = 0;
                    float * raw = wav_from_bytes(wav_bytes, &ns);
                    if (raw && ns > 0) {
                        local_ref_audio.assign(raw, raw + ns);
                        free(raw);
                        p.ref_audio_24k = local_ref_audio.data();
                        p.ref_n_samples = ns;
                        has_local_ref   = true;
                    }
                }
                if (req.ov_ref_text.has_value()) {
                    local_ref_text = req.ov_ref_text.value();
                    p.ref_text     = local_ref_text.c_str();
                }
            }

            if (!has_local_ref) {
                // Fall back to server-wide reference
                if (!server_ref_audio.empty()) {
                    p.ref_audio_24k = server_ref_audio.data();
                    p.ref_n_samples = server_ref_n_samples;
                    p.ref_text      = server_ref_text.c_str();
                } else if (!server_ref_tokens.empty()) {
                    p.ref_audio_tokens = server_ref_tokens.data();
                    p.ref_T            = server_ref_T;
                    p.ref_text         = server_ref_text.c_str();
                }
            }

            // Dump directory (server-wide only)
            p.dump_dir = dump_dir;

            // Streaming callback: forward to httplib sink
            const tts_sink * sink_ptr = &sink;
            p.on_chunk = [](const float * s, int ns, void * u) -> bool {
                return (*static_cast<const tts_sink *>(u))(s, ns);
            };
            p.on_chunk_user_data = (void *) sink_ptr;

            // Synthesize
            struct ov_audio out = {};
            enum ov_status rc   = ov_synthesize(ov, &p, &out);
            ov_audio_free(&out);

            if (rc != OV_STATUS_OK) {
                err = ov_last_error();
                return (int) rc;
            }
            return 0;
        };

        // GET /props : expose server configuration
        be.get_props = [model_path, codec_path, prompt_lang, prompt_instruct,
                        prompt_duration_sec, prompt_denoise, preprocess_prompt,
                        chunk_duration_sec, chunk_threshold_sec,
                        seed_arg, use_fa, clamp_fp16, wav_fmt, dump_dir, device_arg,
                        threads_arg, numa_arg, &cfg]() -> std::string {
            yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
            yyjson_mut_val * root = yyjson_mut_obj(doc);
            yyjson_mut_doc_set_root(doc, root);

            yyjson_mut_obj_add_str(doc, root, "model", basename_of(model_path).c_str());
            yyjson_mut_obj_add_str(doc, root, "codec", basename_of(codec_path).c_str());
            yyjson_mut_obj_add_str(doc, root, "omnivoice_version", OMNIVOICE_VERSION);
            yyjson_mut_obj_add_str(doc, root, "host", cfg.host.c_str());
            yyjson_mut_obj_add_int(doc, root, "port", cfg.port);
            yyjson_mut_obj_add_int(doc, root, "read_timeout_sec", cfg.read_timeout_sec);
            yyjson_mut_obj_add_int(doc, root, "write_timeout_sec", cfg.write_timeout_sec);
            yyjson_mut_obj_add_int(doc, root, "parallel", cfg.n_parallel);

            if (prompt_lang) {
                yyjson_mut_obj_add_str(doc, root, "lang", prompt_lang);
            }
            if (prompt_instruct) {
                yyjson_mut_obj_add_str(doc, root, "instruct", prompt_instruct);
            }
            yyjson_mut_obj_add_real(doc, root, "duration_sec", (double) prompt_duration_sec);
            yyjson_mut_obj_add_bool(doc, root, "denoise", prompt_denoise);
            yyjson_mut_obj_add_bool(doc, root, "preprocess_prompt", preprocess_prompt);
            yyjson_mut_obj_add_real(doc, root, "chunk_duration_sec", (double) chunk_duration_sec);
            yyjson_mut_obj_add_real(doc, root, "chunk_threshold_sec", (double) chunk_threshold_sec);
            yyjson_mut_obj_add_int(doc, root, "seed", seed_arg);
            yyjson_mut_obj_add_bool(doc, root, "flash_attention", use_fa);
            yyjson_mut_obj_add_bool(doc, root, "clamp_fp16", clamp_fp16);
            if (dump_dir) {
                yyjson_mut_obj_add_str(doc, root, "dump_dir", dump_dir);
            }
            if (device_arg) {
                yyjson_mut_obj_add_str(doc, root, "device", device_arg);
            }
            if (threads_arg > 0) {
                yyjson_mut_obj_add_int(doc, root, "threads", threads_arg);
            }
            if (numa_arg >= 0) {
                static const char * numa_names[] = {
                    "disabled", "distribute", "isolate", "numactl", "mirror"};
                yyjson_mut_obj_add_str(doc, root, "numa", numa_names[numa_arg]);
            }

            const char * fmt_name = (wav_fmt == WAV_S16) ? "wav16" :
                                    (wav_fmt == WAV_S24) ? "wav24" : "wav32";
            yyjson_mut_obj_add_str(doc, root, "format", fmt_name);

            // Per-request override capabilities
            yyjson_mut_val * overrides = yyjson_mut_arr(doc);
            const char * override_fields[] = {
                "lang", "instruct", "duration", "seed", "no_denoise",
                "chunk_duration", "chunk_threshold", "format", "stream",
                "ref_wav_base64", "ref_text", nullptr};
            for (int i = 0; override_fields[i]; i++) {
                yyjson_mut_arr_add_str(doc, overrides, override_fields[i]);
            }
            yyjson_mut_obj_add_val(doc, root, "per_request_overrides", overrides);

            char * json = yyjson_mut_write(doc, 0, NULL);
            std::string result = json ? json : "{}";
            if (json) {
                free(json);
            }
            yyjson_mut_doc_free(doc);
            return result;
        };

        int rc = tts_server_run(be, cfg);
        ov_free(ov);
        return rc;

    } catch (const std::exception & e) {
        fprintf(stderr, "[Server] FATAL: %s\n", e.what());
        return 1;
    }
}
