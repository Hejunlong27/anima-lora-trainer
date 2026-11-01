const $ = (id) => document.getElementById(id);
const state = { polling: null, lastLogLen: 0 };

function activateTab(name) {
  const t = document.querySelector(`.tab[data-tab="${name}"]`);
  if (!t) return;
  document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
  document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
  t.classList.add("active");
  $("tab-" + name).classList.add("active");
  if (name === "out") loadOutputs();
  if (name === "mat") { prefillMate(); refreshGpu(); checkComfy(); }
  if (name === "comfy") checkComfyFrame();
  history.replaceState(null, "", "#" + name);
}

document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "out") loadOutputs();
    history.replaceState(null, "", "#" + t.dataset.tab);
  });
});

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

async function loadDefaults() {
  const d = await api("/api/defaults");
  const sel = $("dataset-select");
  sel.innerHTML = "";
  (d.datasets || []).forEach((p) => {
    const o = document.createElement("option");
    o.value = p; o.textContent = p.split(/[\\/]/).pop();
    sel.appendChild(o);
  });
  if (!d.datasets || !d.datasets.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "（未发现数据集目录）";
    sel.appendChild(o);
  }
  sel.onchange = () => { $("dataset-path").value = sel.value; };
  if (d.datasets && d.datasets.length) $("dataset-path").value = d.datasets[0];
  if (d.datasets && d.datasets.length) $("s-path").value = d.datasets[0];
  window.__paths = d.paths || {};
  if (d.latest_infer) {
    $("i-result").innerHTML = `<img src="/api/image?path=${encodeURIComponent(d.latest_infer)}" alt="最近一次出图">`;
    const pth = $("i-path");
    if (pth) pth.textContent = d.latest_infer;
  }
  fillLoras(d.outputs || []);
  const miss = Object.entries(d.exists || {}).filter(([, v]) => !v).map(([k]) => k);
  if (miss.length) logLocal("警告：以下依赖缺失 " + miss.join(", "));
}

function fillLoras(list) {
  const sel = $("i-lora");
  sel.innerHTML = '<option value="">不使用</option>';
  list.forEach((x) => {
    const o = document.createElement("option");
    o.value = x.path; o.textContent = `${x.name}（${x.mb}MB）`;
    sel.appendChild(o);
  });
}

function logLocal(msg) {
  const el = $("log");
  el.textContent += (el.textContent ? "\n" : "") + `[UI] ${msg}`;
  el.scrollTop = el.scrollHeight;
}

document.querySelectorAll(".chip[data-steps]").forEach((b) => {
  b.addEventListener("click", () => {
    $("steps").value = b.dataset.steps;
    $("dim").value = b.dataset.dim;
    $("resolution").value = b.dataset.res;
  });
});

$("btn-start").addEventListener("click", async () => {
  const path = ($("dataset-path").value || $("dataset-select").value || "").trim();
  if (!path) { logLocal("请先选择数据集目录"); return; }
  $("btn-start").disabled = true;
  const cfg = {
    kind: "train",
    dataset: path,
    name: $("name").value.trim(),
    resolution: $("resolution").value,
    steps: comboVal("steps", "steps-c"),
    save_every: comboVal("save_every", "save_every-c"),
    dim: $("dim").value,
    lr: $("lr").value,
    optimizer: $("optimizer").value,
    seed: seedValue("seed", "seed-rand"),
    trigger: $("trigger").value.trim(),
  };
  const r = await api("/api/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg),
  });
  if (!r.ok) { logLocal(r.error || "启动失败"); $("btn-start").disabled = false; }
  else { $("log").textContent = ""; state.lastLogLen = 0; }
});

$("btn-pause").addEventListener("click", async () => {
  if (!confirm("暂停会终止当前训练进程并保留最近一次保存点（含优化器状态），之后可在「记录」页继续。继续？")) return;
  const r = await api("/api/pause", { method: "POST", body: "{}" });
  alert(r.message || r.error || "已暂停");
});

$("btn-stop").addEventListener("click", async () => {
  const r = await api("/api/stop", { method: "POST", body: "{}" });
  if (!r.ok) logLocal(r.error || "停止失败");
});

async function runInfer() {
  const size = (comboVal("i-size", "i-size-c") || "512x768").split("x");
  const mult = comboVal("i-mult", "i-mult-c") || "1.0";
  const trig = $("i-trigger").value.trim().replace(/,\s*$/, "");
  const body = $("i-prompt").value.trim();
  const prompt = trig ? (body ? `${trig}, ${body}` : trig) : body;
  $("i-result").innerHTML = '<span class="muted">生成中…</span>';
  const cfg = {
    kind: "infer",
    prompt,
    negative: $("i-neg").value.trim(),
    lora: $("i-lora").value,
    multiplier: mult,
    width: parseInt(size[0]) || 512,
    height: parseInt(size[1]) || 768,
    steps: parseInt($("i-steps-n").value) || 20,
    guidance: parseFloat($("i-cfg").value) || 5,
    seed: seedValue("i-seed", "i-seed-rand"),
  };
  const r = await api("/api/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg),
  });
  if (!r.ok) $("i-result").innerHTML = `<span class="muted">${r.error}</span>`;
}
$("btn-infer").addEventListener("click", runInfer);

$("btn-scan").addEventListener("click", async () => {
  const p = $("s-path").value.trim();
  if (!p) return;
  const d = await api("/api/scan?path=" + encodeURIComponent(p));
  const box = $("s-result");
  if (!d.ok) { box.innerHTML = `<div class="warn">${d.error}</div>`; return; }
  let html = `<div class="stat-row">
    <div><span>图片</span><b>${d.images}</b></div>
    <div><span>标注</span><b>${d.captions}</b></div>
    <div><span>缺标注</span><b>${d.missing_caption}</b></div>
    <div><span>含质量词</span><b>${d.quality_word_files}</b></div>
  </div>`;
  if (d.sizes && d.sizes.length) html += `<div class="muted" style="margin-bottom:10px">尺寸：${d.sizes.join(" / ")}</div>`;
  if (d.missing_caption > 0) html += `<div class="warn">有 ${d.missing_caption} 张图缺少同名 .txt 标注，训练会被跳过。</div>`;
  if (d.quality_word_files > 0) html += `<div class="warn">有 ${d.quality_word_files} 个标注包含 masterpiece / best quality 等质量词，MOSS 训练需删除（Anima 可保留）。</div>`;
  (d.samples || []).forEach((s) => {
    html += `<div class="cap-sample"><b>${s.name}</b><br>${s.text.replace(/</g, "&lt;")}</div>`;
  });
  box.innerHTML = html;
});

async function loadOutputs() {
  const d = await api("/api/outputs");
  const box = $("out-list");
  if (!d.outputs.length) { box.innerHTML = '<span class="muted">暂无产物</span>'; return; }
  box.innerHTML = d.outputs.map((x) => `
    <div class="out-item">
      <div><b>${x.name}</b><div class="meta">${x.mb} MB · ${x.mtime}</div></div>
      <button class="chip" data-open="${x.path}">所在文件夹</button>
    </div>`).join("");
  box.querySelectorAll("button[data-open]").forEach((b) => {
    b.addEventListener("click", async () => {
      const p = b.dataset.open.replace(/[^\\/]+$/, "");
      await api("/api/open", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: p }) });
    });
  });
  fillLoras(d.outputs);
}
$("btn-refresh").addEventListener("click", loadOutputs);

const PILL_TEXT = { idle: "空闲", running: "运行中", done: "已完成", failed: "失败", stopped: "已停止", paused: "已暂停" };

async function poll() {
  let d;
  try { d = await api("/api/status"); } catch (e) { return; }
  const s = d.stats || {};
  $("m-gpu").textContent = s.gpu_mem != null ? `${(s.gpu_mem / 1024).toFixed(1)} / ${(s.gpu_total / 1024).toFixed(0)} GB` : "—";
  $("m-util").textContent = s.gpu_util != null ? s.gpu_util + "%" : "—";
  $("m-pw").textContent = s.gpu_power != null ? s.gpu_power + " W" : "—";
  $("m-temp").textContent = s.gpu_temp != null ? s.gpu_temp + " °C" : "—";
  $("m-ram").textContent = s.ram_used != null ? `${s.ram_used} / ${s.ram_total} GB` : "—";

  const pill = $("pill");
  pill.textContent = PILL_TEXT[d.status] || d.status;
  pill.className = "pill " + d.status;

  const p = d.progress || {};
  const total = p.total || 0, cur = p.cur || 0;
  $("bar-fill").style.width = total ? Math.min(100, (cur / total) * 100) + "%" : "0%";
  $("p-step").textContent = total ? `${cur} / ${total}` : "—";
  $("p-speed").textContent = p.speed || "—";
  $("p-loss").textContent = p.loss || "—";
  $("p-eta").textContent = p.eta || "—";

  const running = d.status === "running";
  $("btn-start").disabled = running;
  $("btn-stop").disabled = !running;
  $("btn-pause").disabled = !(running && d.job === "train");
  $("btn-infer").disabled = running;

  if (!d.log || !d.log.length) {
    $("log").textContent = "（暂无日志：开始训练后会在这里实时滚动）";
    $("i-log").textContent = "（暂无日志）";
  }
  if (d.log && d.log.length) {
    const el = $("log");
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.textContent = d.log.join("\n");
    if (atBottom) el.scrollTop = el.scrollHeight;
    if ($("tab-infer").classList.contains("active")) {
      const il = $("i-log");
      il.textContent = d.log.slice(-40).join("\n");
      il.scrollTop = il.scrollHeight;
    }
  }

  if (d.job === "infer" && d.result) {
    $("i-result").innerHTML = `<img src="/api/image?path=${encodeURIComponent(d.result)}&t=${Date.now()}" alt="推理结果">`;
    const pth = $("i-path");
    if (pth) pth.textContent = d.result;
  }
  if (d.job === "train" && !running && d.status === "done") loadOutputs();

  const mp = $("m-pill");
  if (mp) {
    mp.textContent = PILL_TEXT[d.status] || d.status;
    mp.className = "pill " + d.status;
    $("m-bar").style.width = total ? Math.min(100, (cur / total) * 100) + "%" : "0%";
    if (d.log && d.log.length) {
      const ml = $("m-log");
      const atB = ml.scrollHeight - ml.scrollTop - ml.clientHeight < 40;
      ml.textContent = d.log.slice(-120).join("\n");
      if (atB) ml.scrollTop = ml.scrollHeight;
    }
  }
}

loadDefaults().then(() => {
  const h = (location.hash || "").replace("#", "");
  if (h) activateTab(h);
  poll();
  setInterval(poll, 1500);
});

function prefillMate() {
  const base = ($("dataset-path").value || "").trim().replace(/\\/g, "/");
  if (!base) return;
  const parent = base.split("/").slice(0, -1).join("/");
  const name = base.split("/").pop();
  if (!$("c-dst").value) $("c-dst").value = `${parent}/${name}_raw`;
  if (!$("c-src").value) $("c-src").value = `${parent}/${name}_raw`;
  if (!$("p-src").value) $("p-src").value = `${parent}/${name}_raw`;
  if (!$("p-dst").value) $("p-dst").value = base;
  if (!$("t-dir").value) $("t-dir").value = base;
}
document.querySelector('.tab[data-tab="mat"]').addEventListener("click", prefillMate);

async function startTask(cfg, btn) {
  if (btn) btn.disabled = true;
  const r = await api("/api/start", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg),
  });
  if (btn) setTimeout(() => { btn.disabled = false; }, 1500);
  if (!r.ok) alert(r.error || "启动失败");
  return r;
}

$("btn-scrape").onclick = (e) => {
  const q = $("c-query").value.trim();
  if (!q) return alert("请填关键词");
  const dst = $("c-dst").value.trim();
  if (!dst) return alert("请填保存目录");
  startTask({ kind: "scrape", query: q, count: $("c-count").value, min_side: $("c-minside").value, dst, prefix: "img" }, e.target);
};

$("btn-urls").onclick = (e) => {
  const urls = $("c-urls").value.trim();
  const dst = $("c-dst").value.trim();
  if (!urls || !dst) return alert("请填直链和保存目录");
  startTask({ kind: "urls", urls, dst, min_side: $("c-minside").value, prefix: "img" }, e.target);
};

$("btn-import").onclick = (e) => {
  const src = $("c-src").value.trim();
  const dst = $("c-dst").value.trim();
  if (!src || !dst) return alert("请填源目录和保存目录");
  startTask({ kind: "import", src, dst }, e.target);
};

$("btn-process").onclick = (e) => {
  const src = $("p-src").value.trim();
  if (!src) return alert("请填源目录");
  const dst = $("p-dst").value.trim() || (src.replace(/[\\/]+$/, "") + "_processed");
  startTask({
    kind: "process", src, dst,
    size: $("p-size").value, blur_th: $("p-blur").value, dup_th: $("p-dup").value,
    crop_face: $("p-face").checked, white_bg: true,
  }, e.target);
};

$("btn-caption").onclick = (e) => {
  const dir = $("t-dir").value.trim() || $("p-dst").value.trim();
  if (!dir) return alert("请填素材目录");
  startTask({
    kind: "caption", dir, mode: $("t-mode").value,
    trigger: $("t-trigger").value.trim(), recaption: $("t-recap").checked,
  }, e.target);
};

async function refreshGpu() {
  try {
    const d = await api("/api/gpu");
    const used = (d.stats && d.stats.gpu_mem) || 0;
    const note = document.querySelector(".notice");
    const txt = $("gpu-text");
    if (used > 1500) {
      const who = (d.holders || []).map((h) => `${h.name}(PID ${h.pid})`).join("、") || "未知";
      txt.textContent = `显存被占用 ${(used / 1024).toFixed(1)} GB（${who}）— 训练与打标前需释放`;
      note.classList.remove("free");
      $("btn-release").style.display = "";
    } else {
      txt.textContent = `显存空闲 ${(used / 1024).toFixed(1)} GB / 8 GB，可以开始训练或打标`;
      note.classList.add("free");
      $("btn-release").style.display = "none";
    }
  } catch (e) { /* ignore */ }
}

$("btn-release").onclick = async () => {
  if (!confirm("将强制关闭占用显存的 ComfyUI 进程（未保存的工作流会丢失）。继续？")) return;
  const r = await api("/api/release_gpu", { method: "POST", body: "{}" });
  if (r.ok && r.killed && r.killed.length) alert("已关闭 PID: " + r.killed.join(", "));
  else alert("没有找到可关闭的 ComfyUI 进程");
  refreshGpu();
};

document.querySelector('.tab[data-tab="mat"]').addEventListener("click", refreshGpu);
setInterval(() => {
  if (document.getElementById("tab-mat").classList.contains("active")) refreshGpu();
}, 6000);

let refPath = "";
async function uploadRef(file) {
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file, file.name);
  const r = await fetch("/api/upload", { method: "POST", body: fd }).then((x) => x.json());
  if (!r.ok) { alert(r.error || "上传失败"); return; }
  refPath = r.path;
  const img = $("g-preview");
  img.src = "/api/image?path=" + encodeURIComponent(refPath) + "&t=" + Date.now();
  img.hidden = false;
  $("g-hint").style.display = "none";
  if (!$("g-name").value) {
    const base = (file.name || "character").replace(/\.[^.]+$/, "");
    $("g-name").value = base.replace(/[^A-Za-z0-9_\-]+/g, "_").slice(0, 24) || "character";
  }
}

$("g-drop").onclick = () => $("g-file").click();
$("g-file").onchange = (e) => uploadRef(e.target.files[0]);
["dragenter", "dragover"].forEach((ev) => $("g-drop").addEventListener(ev, (e) => {
  e.preventDefault(); $("g-drop").classList.add("over");
}));
["dragleave", "drop"].forEach((ev) => $("g-drop").addEventListener(ev, (e) => {
  e.preventDefault(); $("g-drop").classList.remove("over");
}));
$("g-drop").addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) uploadRef(f);
});

async function checkComfy() {
  try {
    const c = await api("/api/comfy");
    const t = $("g-comfy");
    t.textContent = c.online ? "ComfyUI 在线" : "ComfyUI 未启动";
    t.className = "tag " + (c.online ? "ok" : "bad");
  } catch (e) { /* ignore */ }
}
document.querySelector('.tab[data-tab="mat"]').addEventListener("click", checkComfy);

function rollSeed() {
  const v = Math.floor(Math.random() * 2147483646) + 1;
  $("g-seed").value = v;
  return v;
}

$("btn-roll").onclick = (e) => { e.preventDefault(); rollSeed(); };
$("g-rand").onchange = () => {
  const on = $("g-rand").checked;
  $("g-seed").disabled = on;
  if (on) rollSeed();
};

$("btn-genmat").onclick = async (e) => {
  if (!refPath) return alert("请先选择一张参考图");
  const name = ($("g-name").value.trim() || "character_new");
  const seed = $("g-rand").checked ? rollSeed() : ($("g-seed").value || "20261101");
  e.target.disabled = true;
  const r = await api("/api/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      kind: "genmat", image: refPath, name,
      preset: $("g-preset").value, trigger: $("g-trigger").value.trim(),
      resolution: $("g-res").value, seed,
      max_shots: $("g-max").value, resume: $("g-resume").checked,
      dst: "", 
    }),
  });
  setTimeout(() => { e.target.disabled = false; }, 2000);
  if (!r.ok) alert(r.error || "启动失败");
};

function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 600, h = cv.clientHeight || 300;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawLossChart(d) {
  const cv = $("loss-chart");
  if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv);
  ctx.clearRect(0, 0, w, h);
  const L = 58, R = 16, T = 14, B = 30;
  const xs = d.steps, ys = d.loss;
  const x0 = xs[0], x1 = xs[xs.length - 1] || (x0 + 1);
  let lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
  if (d.avg && d.avg.length) {
    lo = Math.min(lo, Math.min.apply(null, d.avg));
    hi = Math.max(hi, Math.max.apply(null, d.avg));
  }
  if (hi - lo < 1e-6) hi = lo + 0.01;
  const pad = (hi - lo) * 0.1;
  lo -= pad; hi += pad;
  const X = (v) => L + (x1 === x0 ? 0 : (v - x0) / (x1 - x0)) * (w - L - R);
  const Y = (v) => T + (hi - v) / (hi - lo) * (h - T - B);

  ctx.strokeStyle = "rgba(0,0,0,0.07)";
  ctx.fillStyle = "#6b6b66";
  ctx.font = "11px -apple-system, 'Segoe UI', sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4;
    const y = Math.round(Y(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(w - R, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(v.toFixed(3), L - 8, y);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const v = x0 + (x1 - x0) * i / 4;
    const x = Math.round(X(v)) + 0.5;
    ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, h - B); ctx.stroke();
    ctx.fillText(String(Math.round(v)), x, h - B + 6);
  }
  ctx.save();
  ctx.translate(14, (h - B + T) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("训练损失", 0, 0);
  ctx.restore();

  function line(px, py, color, width, alpha) {
    if (!px.length) return;
    ctx.globalAlpha = alpha || 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < px.length; i++) {
      const x = X(px[i]), y = Y(py[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  if (d.avg_steps && d.avg_steps.length) line(d.avg_steps, d.avg, "#BA7517", 2, 0.55);
  line(xs, ys, "#185FA5", 1.2, 0.9);

  const lx = X(xs[xs.length - 1]), ly = Y(ys[ys.length - 1]);
  ctx.fillStyle = "#185FA5";
  ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();
}

async function refreshLoss() {
  const sel = $("l-run");
  const run = sel ? sel.value : "";
  let d;
  try { d = await api("/api/loss" + (run ? "?run=" + encodeURIComponent(run) : "")); }
  catch (e) { return; }
  const meta = $("l-meta");
  if (!d.ok) {
    meta.textContent = d.error || "暂无数据";
    const cv = $("loss-chart");
    if (cv) { const { ctx, w, h } = fitCanvas(cv); ctx.clearRect(0, 0, w, h); }
    return;
  }
  if (sel && d.runs && sel.dataset.filled !== String(d.runs.length)) {
    const cur = sel.value;
    sel.innerHTML = '<option value="">最新一次</option>' +
      d.runs.map((r) => `<option value="${r}">${r}</option>`).join("");
    sel.value = cur;
    sel.dataset.filled = String(d.runs.length);
  }
  meta.textContent = `${d.run} · 第 ${d.last_step} 步 · 当前 ${d.last_loss} · 最低 ${d.min_loss} · ${d.count} 点`;
  drawLossChart(d);
}

if ($("l-run")) {
  $("l-run").onchange = refreshLoss;
  setInterval(() => {
    if (document.getElementById("tab-train").classList.contains("active")) refreshLoss();
  }, 6000);
  refreshLoss();
}

async function loadRuns() {
  const box = $("run-list");
  if (!box) return;
  let d;
  try { d = await api("/api/runs"); } catch (e) { return; }
  const runs = d.runs || [];
  const done = runs.filter((r) => r.status === "完成").length;
  $("r-sum").textContent = `共 ${runs.length} 次 · 有产物 ${done} · 其他 ${runs.length - done}`;
  if (!runs.length) { box.innerHTML = '<span class="muted">还没有训练记录</span>'; return; }
  const pill = { "完成": "done", "完成·无参数记录": "done", "已训练·无产物": "stopped", "仅配置": "idle" };
  box.innerHTML = runs.map((r) => {
    const loss = (r.loss && r.loss.last_step)
      ? `第 ${r.loss.last_step} 步 · loss ${r.loss.last_loss} · 最低 ${r.loss.min_loss}`
      : "无损失日志";
    const dur = r.duration_s ? `${Math.max(1, Math.round(r.duration_s / 60))} 分钟` : "—";
    const params = r.steps_cfg
      ? `${r.steps_cfg} 步 · rank ${r.dim || "—"} · lr ${r.lr || "—"} · ${r.optimizer || "—"} · seed ${r.seed || "—"} · 触发词 ${r.trigger || "—"}`
      : "无参数记录（由命令行脚本训练）";
    return `<div class="run-item">
      <div class="run-main">
        <div class="run-title"><b>${r.name}</b>
          <span class="pill ${pill[r.status] || "idle"}">${r.status}</span>
          ${r.bundle ? '<span class="tag-mini">Anima</span>' : ""}
        </div>
        <div class="run-meta">${r.started_at} · 数据集 ${r.dataset}${r.images ? `（${r.images} 张）` : ""} · ${r.resolution || "—"} px · 耗时 ${dur}</div>
        <div class="run-params">${params}</div>
      </div>
      <div class="run-side">
        <div class="run-loss">${loss}</div>
        <div class="run-out">${r.output.exists ? r.output.mb + " MB" : "无产物"}</div>
        ${r.output.exists ? `<button class="chip" data-try="${r.name}">试跑</button>` : ""}
        ${(r.state || r.output.exists) ? `<button class="chip" data-resume="${r.name}">继续训练</button>` : ""}
        <button class="chip" data-dir="${r.name}">打开目录</button>
      </div>
    </div>`;
  }).join("");
  window.__runs = runs;
  box.querySelectorAll("button[data-try]").forEach((b) => {
    b.onclick = () => inferFromRun(b.dataset.try);
  });
  box.querySelectorAll("button[data-resume]").forEach((b) => {
    b.onclick = async () => {
      const name = b.dataset.resume;
      if (!confirm(`从「${name}」最近一次保存点继续训练到原定步数？`)) return;
      b.disabled = true;
      const r = await api("/api/resume", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      b.disabled = false;
      if (r.ok) { alert("已开始续训，可在「训练」页查看进度"); activateTab("train"); }
      else alert(r.error || "续训失败");
      loadRuns();
    };
  });
  box.querySelectorAll("button[data-dir]").forEach((b) => {
    b.onclick = () => api("/api/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: "D:/anima-train/runs/" + b.dataset.dir }),
    });
  });
}

if ($("btn-runs")) {
  $("btn-runs").onclick = loadRuns;
  document.querySelector('.tab[data-tab="log"]').addEventListener("click", loadRuns);
  loadRuns();
}

function inferFromRun(name) {
  const r = (window.__runs || []).find((x) => x.name === name);
  if (!r) return;
  if (!r.output || !r.output.exists) { alert("该任务没有可用产物"); return; }
  $("i-trigger").value = r.trigger || "";
  $("i-prompt").value = "masterpiece, best quality, full body, standing, looking at viewer, simple background";
  $("i-seed").value = r.seed || 42;
  const sel = $("i-lora");
  sel.value = "";
  for (const o of sel.options) {
    if (o.value && o.value.endsWith("\\" + name + ".safetensors")) { sel.value = o.value; break; }
  }
  activateTab("infer");
  $("i-result").innerHTML = '<span class="muted">已填入触发词与 LoRA，正在生成…</span>';
  runInfer();
}

async function checkTrigger() {
  const ds = ($("dataset-path").value || "").trim();
  const tg = ($("trigger").value || "").trim();
  const info = $("trigger-info");
  if (!ds) { info.textContent = "请先选择数据集目录"; return; }
  const d = await api("/api/trigger_check?dataset=" + encodeURIComponent(ds) + "&trigger=" + encodeURIComponent(tg));
  if (!d.ok) { info.textContent = d.error || "检查失败"; return; }
  if (!tg) {
    info.textContent = `该数据集共 ${d.total} 条标注；未填触发词，仅做统计。`;
    return;
  }
  if (d.missing === 0) {
    info.innerHTML = `<span style="color:#3b6d11">✓ ${d.total} 条标注全部包含「${tg}」，可以直接训练。</span>`;
  } else {
    info.innerHTML = `<span style="color:#a32d2d">✗ ${d.total} 条标注中有 ${d.missing} 条不含「${tg}」</span> ` +
      `<button id="btn-trigger-fix" class="chip mini">补全前缀</button>`;
    const btn = $("btn-trigger-fix");
    if (btn) btn.onclick = async () => {
      if (!confirm(`将给 ${d.missing} 条缺少触发词的标注加上「${tg}, 」前缀。\n原文件会先备份到 dataset 下的 _caption_backup_* 目录。继续？`)) return;
      btn.disabled = true;
      const a = await api("/api/trigger_apply", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset: ds, trigger: tg }),
      });
      if (a.ok) { alert(`已补全 ${a.updated} 条，备份在 ${a.backup || "（无改动）"}`); checkTrigger(); }
      else alert(a.error || "补全失败");
    };
  }
}
if ($("btn-trigger")) $("btn-trigger").onclick = checkTrigger;

$("btn-infer-dir").onclick = () => {
  const dir = (window.__paths && window.__paths.infer) || "";
  if (!dir) { alert("尚未获取到出图目录，请刷新页面"); return; }
  api("/api/open", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: dir }),
  });
};

function comboVal(id, cid) {
  const s = $(id);
  if (!s) return "";
  if (s.value === "__custom") {
    const c = $(cid);
    return c && c.value.trim() ? c.value.trim() : "";
  }
  return s.value;
}

function comboBind(id, cid) {
  const s = $(id), c = $(cid);
  if (!s || !c) return;
  s.addEventListener("change", () => {
    c.hidden = s.value !== "__custom";
    if (!c.hidden) c.focus();
  });
}

function rollSeedTo(id) {
  const v = Math.floor(Math.random() * 2147483646) + 1;
  $(id).value = v;
  return v;
}

function seedValue(inputId, randId) {
  const r = $(randId);
  return r && r.checked ? rollSeedTo(inputId) : ($(inputId).value || "42");
}

[["steps", "steps-c"], ["save_every", "save_every-c"],
 ["i-size", "i-size-c"], ["i-mult", "i-mult-c"]].forEach(([a, b]) => comboBind(a, b));

if ($("btn-seed-roll")) $("btn-seed-roll").onclick = () => rollSeedTo("seed");
if ($("btn-i-roll")) $("btn-i-roll").onclick = () => rollSeedTo("i-seed");
if ($("seed-rand")) $("seed-rand").onchange = () => {
  $("seed").disabled = $("seed-rand").checked;
  if ($("seed-rand").checked) rollSeedTo("seed");
};
if ($("i-seed-rand")) $("i-seed-rand").onchange = () => {
  $("i-seed").disabled = $("i-seed-rand").checked;
  if ($("i-seed-rand").checked) rollSeedTo("i-seed");
};

async function checkComfyFrame() {
  const note = $("comfy-note"), st = $("comfy-state"), fr = $("comfy-frame");
  if (!st) return;
  let d;
  try { d = await api("/api/comfy"); } catch (e) { return; }
  if (d.online) {
    note.classList.add("free");
    st.textContent = "已连接 ComfyUI（127.0.0.1:8188）";
    if (fr.dataset.loaded !== "1") {
      fr.src = "http://127.0.0.1:8188";
      fr.dataset.loaded = "1";
    }
  } else {
    note.classList.remove("free");
    st.textContent = "ComfyUI 未启动：请先启动 ComfyUI（端口 8188），再点「重新检测」";
    if (fr.dataset.loaded === "1") { fr.src = "about:blank"; fr.dataset.loaded = ""; }
  }
}
if ($("btn-comfy-check")) {
  $("btn-comfy-check").onclick = checkComfyFrame;
  $("btn-comfy-outer").onclick = () => window.open("http://127.0.0.1:8188", "_blank");
  document.querySelector('.tab[data-tab="comfy"]').addEventListener("click", checkComfyFrame);
  checkComfyFrame();
}
