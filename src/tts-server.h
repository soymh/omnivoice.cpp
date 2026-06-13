#pragma once
// tts-server.h: shared OpenAI-compatible TTS HTTP core for the *.cpp ports.
//
// One synthesis context lives GPU resident for the process lifetime. The
// project tool fills a tts_backend adapter that wires its own ABI
// (qt_synthesize / ov_synthesize) into the generic sink, then calls
// tts_server_run. The HTTP layer, tuning, OAI parsing and audio framing
// are identical across projects ; only the adapter differs.
//
// Endpoints:
//   POST /v1/audio/speech   OAI text-to-speech
//   GET  /v1/models         single loaded model
//   GET  /v1/voices         named speakers (empty when the model has none)
//   GET  /health            liveness probe
//
// Audio out: response_format "pcm" streams s16le 24 kHz mono chunked as it
// is generated (real time), "wav" returns a one-shot RIFF file. pcm is the
// default so streaming is on unless the client asks for a file.

#include "../vendor/cpp-httplib/httplib.h"
#include "audio-io.h"
#include "yyjson.h"

#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

// One synthesis request parsed from the OAI JSON body.
struct tts_request {
    std::string input;         // text to speak
    std::string voice;         // OAI voice, mapped to a speaker by the adapter
    std::string instructions;  // OAI instructions, mapped to the ABI instruct field
    std::string format;        // "pcm" (stream) or "wav" (one-shot)
    float       speed;         // OAI speed, parsed then ignored (no time stretch in the ABI)

    // Omnivoice extension fields (optional, parsed from JSON body)
    std::optional<std::string> ov_lang;
    std::optional<std::string> ov_instruct;
    std::optional<float>       ov_duration_sec;
    std::optional<int>         ov_seed;
    std::optional<bool>        ov_no_denoise;
    std::optional<float>       ov_chunk_duration_sec;
    std::optional<float>       ov_chunk_threshold_sec;
    std::optional<std::string> ov_format;          // "wav16" / "wav24" / "wav32"
    std::optional<bool>        ov_stream;          // stream chunks as they're produced
    std::optional<std::string> ov_ref_wav_base64;  // base64-encoded WAV for per-request voice cloning
    std::optional<std::string> ov_ref_text;        // transcript for per-request reference
};

// The adapter pushes mono f32 24 kHz audio here. Returns false to abort the
// synthesis (client gone or cancellation), which propagates into the ABI
// on_chunk and stops generation.
using tts_sink = std::function<bool(const float * samples, int n_samples)>;

// Adapter implemented by each project tool.
struct tts_backend {
    std::string              model_id;  // reported by GET /v1/models
    std::vector<std::string> voices;    // reported by GET /v1/voices, may be empty
    // Run synthesis. When the request streams, the adapter routes the ABI
    // on_chunk to sink ; otherwise it pushes the whole buffer once. Returns
    // the ABI status (0 on success), and fills err with the ABI message on
    // failure. The shared layer maps the status to an HTTP code.
    std::function<int(const tts_request & req, const tts_sink & sink, std::string & err)> synthesize;
    // Returns a JSON string describing server properties. Called by GET /props.
    std::function<std::string()> get_props;
    // Default WAV output format. Overridable per-request via ov_format.
    WavFormat wav_fmt = WAV_S16;
};

struct server_config {
    std::string host = "127.0.0.1";
    int         port = 8080;
    int         read_timeout_sec = 60;
    int         write_timeout_sec = 120;
    int         n_parallel = 1;
};

// Single GPU context : synthesis is serialised FIFO across connections.
static std::mutex        g_synth_mutex;
static httplib::Server * g_svr = nullptr;

static void tts_on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

// Clamp to [-1, 1] and scale to s16. lrintf rounds to nearest, ties to even.
static inline int16_t tts_f32_to_s16(float x) {
    float v = x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
    return (int16_t) lrintf(v * 32767.0f);
}

// Append a mono f32 block as s16le bytes onto out.
static void tts_append_s16le(std::string & out, const float * samples, int n_samples) {
    size_t base = out.size();
    out.resize(base + (size_t) n_samples * 2);
    char * p = &out[base];
    for (int i = 0; i < n_samples; i++) {
        int16_t s = tts_f32_to_s16(samples[i]);
        *p++      = (char) ((uint16_t) s & 0xff);
        *p++      = (char) (((uint16_t) s >> 8) & 0xff);
    }
}

// Write a JSON error body in the OAI error envelope and set the status.
static void tts_json_error(httplib::Response & res, int status, const char * type, const char * message) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * err = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, err, "message", message);
    yyjson_mut_obj_add_str(doc, err, "type", type);
    yyjson_mut_obj_add_val(doc, root, "error", err);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.status  = status;
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

// Parse the OAI body into req. Returns false and fills err on bad input.
static bool tts_parse_request(const std::string & body, tts_request & req, std::string & err) {
    yyjson_doc * doc = yyjson_read(body.c_str(), body.size(), 0);
    if (!doc) {
        err = "request body is not valid JSON";
        return false;
    }
    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        err = "request body must be a JSON object";
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_val * input = yyjson_obj_get(root, "input");
    if (!yyjson_is_str(input) || yyjson_get_len(input) == 0) {
        err = "'input' must be a non-empty string";
        yyjson_doc_free(doc);
        return false;
    }
    req.input = yyjson_get_str(input);

    yyjson_val * voice = yyjson_obj_get(root, "voice");
    req.voice          = yyjson_is_str(voice) ? yyjson_get_str(voice) : "";

    yyjson_val * instructions = yyjson_obj_get(root, "instructions");
    req.instructions          = yyjson_is_str(instructions) ? yyjson_get_str(instructions) : "";

    yyjson_val * fmt = yyjson_obj_get(root, "response_format");
    req.format       = yyjson_is_str(fmt) ? yyjson_get_str(fmt) : "pcm";

    yyjson_val * speed = yyjson_obj_get(root, "speed");
    req.speed          = yyjson_is_num(speed) ? (float) yyjson_get_num(speed) : 1.0f;

    // Omnivoice extension fields (all optional)
    yyjson_val * ov_lang = yyjson_obj_get(root, "lang");
    if (yyjson_is_str(ov_lang)) {
        req.ov_lang = yyjson_get_str(ov_lang);
    }

    yyjson_val * ov_instruct = yyjson_obj_get(root, "instruct");
    if (yyjson_is_str(ov_instruct)) {
        req.ov_instruct = yyjson_get_str(ov_instruct);
    }

    yyjson_val * ov_dur = yyjson_obj_get(root, "duration");
    if (yyjson_is_num(ov_dur)) {
        req.ov_duration_sec = (float) yyjson_get_num(ov_dur);
    }

    yyjson_val * ov_seed = yyjson_obj_get(root, "seed");
    if (yyjson_is_int(ov_seed)) {
        req.ov_seed = (int) yyjson_get_int(ov_seed);
    }

    yyjson_val * ov_no_denoise = yyjson_obj_get(root, "no_denoise");
    if (yyjson_is_bool(ov_no_denoise)) {
        req.ov_no_denoise = yyjson_get_bool(ov_no_denoise);
    }

    yyjson_val * ov_cd = yyjson_obj_get(root, "chunk_duration");
    if (yyjson_is_num(ov_cd)) {
        req.ov_chunk_duration_sec = (float) yyjson_get_num(ov_cd);
    }

    yyjson_val * ov_ct = yyjson_obj_get(root, "chunk_threshold");
    if (yyjson_is_num(ov_ct)) {
        req.ov_chunk_threshold_sec = (float) yyjson_get_num(ov_ct);
    }

    yyjson_val * ov_fmt = yyjson_obj_get(root, "format");
    if (yyjson_is_str(ov_fmt)) {
        req.ov_format = yyjson_get_str(ov_fmt);
    }

    yyjson_val * ov_stream = yyjson_obj_get(root, "stream");
    if (yyjson_is_bool(ov_stream)) {
        req.ov_stream = yyjson_get_bool(ov_stream);
    }

    yyjson_val * ov_ref = yyjson_obj_get(root, "ref_wav_base64");
    if (yyjson_is_str(ov_ref)) {
        req.ov_ref_wav_base64 = yyjson_get_str(ov_ref);
    }

    yyjson_val * ov_ref_text = yyjson_obj_get(root, "ref_text");
    if (yyjson_is_str(ov_ref_text)) {
        req.ov_ref_text = yyjson_get_str(ov_ref_text);
    }

    yyjson_doc_free(doc);

    if (req.format != "pcm" && req.format != "wav") {
        err = "response_format must be 'pcm' or 'wav'";
        return false;
    }
    return true;
}

// Map an ABI status to an HTTP code. The two ABIs share numeric values:
// -1 invalid params, -2 mode/instruct invalid -> client error ; the rest
// are server side failures.
static int tts_status_to_http(int rc) {
    if (rc == 0) {
        return 200;
    }
    if (rc == -1 || rc == -2) {
        return 400;
    }
    return 502;
}

// Resolve the WAV output format. Per-request ov_format takes priority;
// otherwise use the backend default.
static WavFormat tts_resolve_wav_fmt(const tts_backend & be, const tts_request & req) {
    if (req.ov_format.has_value()) {
        WavFormat fmt = WAV_S16;
        if (audio_parse_format(req.ov_format->c_str(), fmt)) {
            return fmt;
        }
    }
    return be.wav_fmt;
}

static void tts_handle_speech(const tts_backend & be, const httplib::Request & http_req, httplib::Response & res) {
    tts_request req;
    std::string err;
    if (!tts_parse_request(http_req.body, req, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }

    bool should_stream = (req.format == "pcm") || req.ov_stream.value_or(false);

    if (!should_stream) {
        // One-shot : collect the whole utterance, then emit a RIFF file
        // with the resolved WAV format.
        WavFormat        fmt = tts_resolve_wav_fmt(be, req);
        std::vector<float> buf;
        tts_sink           sink = [&buf](const float * s, int n) {
            buf.insert(buf.end(), s, s + n);
            return true;
        };
        std::string synth_err;
        int         rc;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            rc = be.synthesize(req, sink, synth_err);
        }
        if (rc != 0) {
            tts_json_error(res, tts_status_to_http(rc), "server_error",
                           synth_err.empty() ? "synthesis failed" : synth_err.c_str());
            return;
        }
        std::string wav = audio_encode_wav(buf.data(), (int) buf.size(), 24000, fmt);
        res.set_content(std::move(wav), "audio/wav");
        return;
    }

    // Streaming : run synthesis inside the chunked provider on the connection
    // thread, pushing s16le frames as the codec produces them. A failed
    // sink.write means the client disconnected, which aborts generation and
    // frees the GPU instead of finishing a stream nobody reads.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider("audio/pcm", [&be, req](size_t, httplib::DataSink & sink) mutable -> bool {
        tts_sink push = [&sink](const float * s, int n) {
            std::string bytes;
            tts_append_s16le(bytes, s, n);
            return sink.write(bytes.data(), bytes.size());
        };
        std::string synth_err;
        {
            std::lock_guard<std::mutex> lock(g_synth_mutex);
            be.synthesize(req, push, synth_err);
        }
        sink.done();
        return true;
    });
}

static void tts_handle_models(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "object", "list");
    yyjson_mut_val * data = yyjson_mut_arr(doc);
    yyjson_mut_val * one  = yyjson_mut_obj(doc);
    yyjson_mut_obj_add_str(doc, one, "id", be.model_id.c_str());
    yyjson_mut_obj_add_str(doc, one, "object", "model");
    yyjson_mut_obj_add_str(doc, one, "owned_by", "local");
    yyjson_mut_arr_add_val(data, one);
    yyjson_mut_obj_add_val(doc, root, "data", data);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_voices(const tts_backend & be, const httplib::Request &, httplib::Response & res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_val * arr = yyjson_mut_arr(doc);
    for (const std::string & v : be.voices) {
        yyjson_mut_val * one = yyjson_mut_obj(doc);
        yyjson_mut_obj_add_str(doc, one, "name", v.c_str());
        yyjson_mut_arr_add_val(arr, one);
    }
    yyjson_mut_obj_add_val(doc, root, "voices", arr);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void tts_handle_health(const httplib::Request &, httplib::Response & res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
}

static int tts_server_run(const tts_backend & be, const server_config & cfg) {
    httplib::Server svr;
    g_svr = &svr;

    // per-operation socket idle timeouts. read is small (text in), write is
    // generous to cover a long streamed utterance without tripping on a slow
    // client.
    svr.set_read_timeout(cfg.read_timeout_sec);
    svr.set_write_timeout(cfg.write_timeout_sec);

    // reject oversized bodies. text plus an optional reference clip stays
    // well under this.
    svr.set_payload_max_length(32 * 1024 * 1024);

    // Nagle coalescing holds small packets back for tens of ms ; streamed
    // PCM chunks must leave the socket the moment they are written.
    svr.set_tcp_nodelay(true);

    // SO_REUSEADDR lets us rebind a port still in TIME_WAIT after a restart.
    // SO_REUSEPORT is deliberately not set : a second instance on the same
    // port then fails with EADDRINUSE instead of silently sharing the socket
    // and splitting traffic between two daemons.
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    // permissive CORS so a browser client can call the API directly.
    svr.set_default_headers({
        { "Access-Control-Allow-Origin", "*" }
    });
    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
    });

    svr.Post("/v1/audio/speech",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Post("/synthesize",
             [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Get("/v1/models",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_models(be, req, res); });
    svr.Get("/v1/voices",
            [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voices(be, req, res); });
    svr.Get("/health", tts_handle_health);
    svr.Get("/props", [&be](const httplib::Request &, httplib::Response & res) {
        if (be.get_props) {
            std::string json = be.get_props();
            res.set_content(json, "application/json");
        } else {
            res.set_content("{}", "application/json");
        }
    });

    signal(SIGINT, tts_on_signal);
    signal(SIGTERM, tts_on_signal);

    fprintf(stderr, "[Server] model %s\n", be.model_id.c_str());
    fprintf(stderr, "[Server] listening on %s:%d\n", cfg.host.c_str(), cfg.port);
    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }
    return 0;
}
