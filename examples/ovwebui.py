#!/usr/bin/env python3
"""OmniVoice Web UI — browser interface for omnivoice-server.

Usage:
    python ovwebui.py [--port 5000] [--server-host 127.0.0.1] [--server-port 8080]

Requires: flask (pip install flask)
Open http://localhost:5000 in your browser.
"""

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

from flask import Flask, jsonify, request, Response

SERVER_BASE = "http://127.0.0.1:8080"
app = Flask(__name__)


def _proxy(method, path, body=None, headers=None):
    url = f"{SERVER_BASE}{path}"
    hdrs = {}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        resp = urllib.request.urlopen(req)
        return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return 502, json.dumps({"error": str(e.reason)}).encode()


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/api/health")
def api_health():
    status, data = _proxy("GET", "/health")
    return Response(data, status=status, mimetype="application/json")


@app.route("/api/props")
def api_props():
    status, data = _proxy("GET", "/props")
    return Response(data, status=status, mimetype="application/json")


@app.route("/api/synthesize", methods=["POST"])
def api_synthesize():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    body = {"input": text, "response_format": "wav"}

    for key in ("lang", "instruct", "format", "ref_text"):
        val = request.form.get(key)
        if val:
            body[key] = val

    for key in ("duration", "seed", "chunk_duration", "chunk_threshold"):
        val = request.form.get(key)
        if val:
            try:
                body[key] = float(val) if "." in val else int(val)
            except ValueError:
                pass

    if request.form.get("no_denoise") == "true":
        body["no_denoise"] = True

    if "ref_wav" in request.files:
        f = request.files["ref_wav"]
        if f and f.filename:
            body["ref_wav_base64"] = base64.b64encode(f.read()).decode("ascii")

    body = {k: v for k, v in body.items() if v != ""}

    json_body = json.dumps(body).encode("utf-8")
    status, data = _proxy("POST", "/v1/audio/speech", json_body,
                          {"Content-Type": "application/json"})

    if status != 200:
        try:
            err = json.loads(data).get("error", {}).get("message", str(data[:200]))
        except Exception:
            err = str(data[:200])
        return jsonify({"error": err}), status

    return Response(data, mimetype="audio/wav")


# ---------------------------------------------------------------------------
# HTML template (embedded for zero external files)
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>OmniVoice TTS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#1a1a2e;color:#eaeaea;min-height:100vh
}
.container{max-width:680px;margin:0 auto;padding:24px}

header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}
header h1{
  font-size:22px;font-weight:700;
  background:linear-gradient(135deg,#e94560,#533483);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text
}
#status{font-size:12px;padding:3px 10px;border-radius:10px;font-weight:500}
#status.ok{background:#1a5c3a;color:#6fcf97}
#status.err{background:#5c1a1a;color:#e74c3c}
#status.wait{background:#2a2a4a;color:#888}

textarea{
  width:100%;min-height:100px;padding:10px 12px;
  background:#16213e;color:#eaeaea;border:1px solid #0f3460;border-radius:6px;
  font-size:14px;resize:vertical;font-family:inherit
}
textarea:focus{outline:none;border-color:#e94560}

.grid{display:flex;flex-direction:column;gap:10px;margin:14px 0}
.row{display:flex;gap:10px;flex-wrap:wrap}
.cell{flex:1;min-width:130px}
.cell label{display:block;font-size:11px;color:#888;margin-bottom:3px;text-transform:uppercase;letter-spacing:.4px}
.cell input,.cell select{
  width:100%;padding:7px 9px;background:#16213e;color:#eaeaea;
  border:1px solid #0f3460;border-radius:5px;font-size:13px;font-family:inherit
}
.cell input:focus,.cell select:focus{outline:none;border-color:#e94560}
.cell.cb{display:flex;align-items:flex-end;padding-bottom:2px}
.cell.cb label{display:flex;align-items:center;gap:5px;font-size:12px;text-transform:none;letter-spacing:0;cursor:pointer}
.cell.cb input{width:auto}

details{
  background:#16213e;border-radius:6px;padding:10px 12px;margin:10px 0;
  border:1px solid #0f3460;font-size:13px
}
details summary{cursor:pointer;color:#888;font-weight:500}
details[open] summary{margin-bottom:10px}

input[type=file]{font-size:12px}

.btn{
  width:100%;padding:12px;
  background:linear-gradient(135deg,#e94560,#533483);
  color:#fff;border:none;border-radius:6px;font-size:15px;font-weight:600;
  cursor:pointer;transition:opacity .2s
}
.btn:hover{opacity:.9}
.btn:disabled{opacity:.5;cursor:not-allowed}

.player{
  margin-top:14px;padding:14px;background:#16213e;border-radius:6px;
  border:1px solid #0f3460
}
.player audio{width:100%}
.player .bar{margin-top:8px;display:flex;gap:8px}
.player .bar a,.player .bar button{
  padding:5px 12px;font-size:12px;border-radius:5px;
  border:1px solid #0f3460;background:#0f3460;color:#eaeaea;
  cursor:pointer;text-decoration:none;transition:background .2s
}
.player .bar a:hover,.player .bar button:hover{background:#533483}

.info{margin-top:20px;padding:14px;background:#16213e;border-radius:6px;border:1px solid #0f3460}
.info h3{font-size:13px;color:#888;margin-bottom:6px}
.info pre{font-size:11px;color:#777;overflow-x:auto;white-space:pre-wrap;font-family:'SF Mono','Fira Code',monospace}

.err{margin-top:10px;padding:8px 12px;background:rgba(231,76,60,.12);border:1px solid #e74c3c;border-radius:5px;color:#e74c3c;font-size:12px;display:none}
</style>
</head>
<body>
<div class=container>
  <header>
    <h1>OmniVoice TTS</h1>
    <span id=status class=wait>● connecting</span>
  </header>

  <textarea id=text placeholder="Enter text to speak..."></textarea>

  <div class=grid>
    <div class=row>
      <div class=cell>
        <label>Language</label>
        <select id=lang>
          <option value="">Auto</option>
          <option value=en>English</option>
          <option value=zh>Chinese</option>
        </select>
      </div>
      <div class=cell>
        <label>Style</label>
        <input id=instruct placeholder="e.g. cheerful">
      </div>
      <div class=cell>
        <label>Duration (sec)</label>
        <input id=duration type=number min=0 step=.5 placeholder="auto">
      </div>
    </div>
    <div class=row>
      <div class=cell>
        <label>Seed</label>
        <input id=seed type=number placeholder="random">
      </div>
      <div class=cell>
        <label>Format</label>
        <select id=fmt>
          <option value="">Default (wav16)</option>
          <option value=wav16>WAV 16-bit</option>
          <option value=wav24>WAV 24-bit</option>
          <option value=wav32>WAV 32-bit float</option>
        </select>
      </div>
      <div class="cell cb">
        <label><input id=noDenoise type=checkbox> No denoise</label>
      </div>
    </div>
  </div>

  <details>
    <summary>Advanced &amp; Voice cloning</summary>
    <div class=row style=margin-bottom:6px>
      <div class=cell><label>Chunk duration</label><input id=chunkDur type=number step=.5 placeholder=15.0></div>
      <div class=cell><label>Chunk threshold</label><input id=chunkThresh type=number step=.5 placeholder=30.0></div>
    </div>
    <div class=row>
      <div class=cell><label>Reference WAV</label><input id=refWav type=file accept=.wav></div>
      <div class=cell><label>Transcript</label><input id=refText placeholder="reference text"></div>
    </div>
  </details>

  <button class=btn id=synthBtn> Synthesize</button>

  <div class=player id=player style=display:none>
    <audio id=audio controls></audio>
    <div class=bar>
      <a id=downloadLink download=omnivoice.wav> Download WAV</a>
      <button id=replayBtn> Play again</button>
    </div>
  </div>

  <div class=info>
    <h3>Server Info</h3>
    <pre id=serverInfo>Loading...</pre>
  </div>

  <div class=err id=error></div>
</div>

<script>
(function(){
const BASE = '';
const $ = id => document.getElementById(id);
const textEl = $('text'), synthBtn = $('synthBtn'), player = $('player');
const audio = $('audio'), dlLink = $('downloadLink'), statusEl = $('status');
const serverInfo = $('serverInfo'), errorEl = $('error');
let lastUrl = null;

function setStatus(ok, msg){
  statusEl.textContent = msg;
  statusEl.className = ok ? 'ok' : 'err';
}

function showError(m){
  errorEl.textContent = m; errorEl.style.display = 'block';
}
function hideError(){ errorEl.style.display = 'none'; }

function setBusy(b){
  synthBtn.disabled = b;
  synthBtn.textContent = b ? ' Synthesizing...' : ' Synthesize';
}

async function healthCheck(){
  try{
    const r = await fetch(BASE+'/api/health');
    setStatus(r.ok, r.ok ? '● online' : '● error');
  }catch(e){ setStatus(false, '● offline'); }
}

async function loadProps(){
  try{
    const r = await fetch(BASE+'/api/props');
    serverInfo.textContent = r.ok ? JSON.stringify(await r.json(),null,2) : 'Failed to load';
  }catch(e){ serverInfo.textContent = 'Server unreachable'; }
}

function showPlayer(blob){
  if(lastUrl) URL.revokeObjectURL(lastUrl);
  const url = URL.createObjectURL(blob);
  lastUrl = url;
  audio.src = url;
  dlLink.href = url;
  player.style.display = 'block';
  audio.play().catch(function(){});
}

synthBtn.addEventListener('click', async function(){
  const text = textEl.value.trim();
  if(!text){ showError('Please enter some text.'); return; }
  hideError(); setBusy(true);

  const f = new FormData();
  f.append('text', text);
  if($('lang').value) f.append('lang', $('lang').value);
  if($('instruct').value) f.append('instruct', $('instruct').value);
  if($('duration').value) f.append('duration', $('duration').value);
  if($('seed').value) f.append('seed', $('seed').value);
  if($('fmt').value) f.append('format', $('fmt').value);
  if($('noDenoise').checked) f.append('no_denoise', 'true');
  if($('chunkDur').value) f.append('chunk_duration', $('chunkDur').value);
  if($('chunkThresh').value) f.append('chunk_threshold', $('chunkThresh').value);
  if($('refText').value) f.append('ref_text', $('refText').value);
  if($('refWav').files[0]) f.append('ref_wav', $('refWav').files[0]);

  try{
    const r = await fetch(BASE+'/api/synthesize', {method:'POST',body:f});
    if(!r.ok){
      const ed = await r.json().catch(function(){return {error:r.statusText};});
      throw new Error(ed.error || 'HTTP '+r.status);
    }
    showPlayer(await r.blob());
  }catch(e){ showError(e.message); }
  finally{ setBusy(false); }
});

$('replayBtn').addEventListener('click', function(){
  if(audio.src){ audio.currentTime=0; audio.play().catch(function(){}); }
});

textEl.addEventListener('keydown', function(e){
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter') synthBtn.click();
});

healthCheck();
loadProps();
setInterval(healthCheck, 20000);
})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OmniVoice Web UI")
    p.add_argument("--host", default="127.0.0.1", help="web UI listen address")
    p.add_argument("--port", type=int, default=5000, help="web UI listen port")
    p.add_argument("--server-host", default="127.0.0.1",
                   help="omnivoice-server host")
    p.add_argument("--server-port", type=int, default=8080,
                   help="omnivoice-server port")
    args = p.parse_args()
    SERVER_BASE = f"http://{args.server_host}:{args.server_port}"
    print(f"[ovwebui] proxying → {SERVER_BASE}")
    print(f"[ovwebui] open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=True)
