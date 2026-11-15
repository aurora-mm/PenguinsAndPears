
import { pickKernel, unpackData, gunzipIfNeeded } from "./engine.js";
import { Tokenizer, controlPrefix } from "./text.js";
import { SafetyFilter, better, rejectionSummary } from "./filters.js";

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !onProgress) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const out = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { out.set(c, off); off += c.length; }
  return out.buffer;
}

const once = (worker, type) =>
  new Promise((resolve, reject) => {
    const on = (e) => { if (e.data.type === type) { worker.removeEventListener("message", on); resolve(e.data); } };
    worker.addEventListener("message", on);
    worker.addEventListener("error", (e) => reject(new Error(e.message || "worker failed to start")), { once: true });
  });

export class Generator {
  static async load(options = {}, onProgress = () => {}) {
    const t0 = performance.now();
    const base = options.base || new URL("../dist/", import.meta.url);
    const kind = "f16";
    const [dataBuf, weightsRaw, plain, relaxed] = await Promise.all([
      fetchBytes(new URL("data.pack", base)),
      fetchBytes(new URL(`model.${kind}.bin`, base), (got, total) => onProgress("download", got, total)),
      fetchBytes(new URL("kernel.wasm", base)),
      fetchBytes(new URL("kernel-relaxed.wasm", base)),
    ]);
    const downloadMs = performance.now() - t0;
    onProgress("start");
    const [meta, weights] = await Promise.all([unpackData(dataBuf), gunzipIfNeeded(weightsRaw)]);
    const pick = options.kernel === "plain" ? { bytes: plain, relaxed: false }
      : pickKernel(new Uint8Array(plain), new Uint8Array(relaxed));
    const module = await WebAssembly.compile(pick.bytes);

    const self = new Generator();
    self.meta = meta;
    self.tok = new Tokenizer(meta);
    self.safety = new SafetyFilter(meta.safety);
    self.info = { kind, relaxed: pick.relaxed, downloadMs, bytes: dataBuf.byteLength + weightsRaw.byteLength + plain.byteLength };

    const cores = navigator.hardwareConcurrency || 4;
    const n = Math.max(1, Math.min(options.workers || Math.ceil(cores / 2), 16));
    self.slots = Math.max(1, Math.min(options.slots || Math.ceil(meta.generation.candidates / n), 48));
    const slim = { model: meta.model, vocab: { length: meta.vocab.length }, ids: meta.ids, forbidden: meta.forbidden, filler: meta.filler };

    self.filter = new Worker(new URL("./filter-worker.js", import.meta.url), { type: "module" });
    const filterReady = once(self.filter, "ready");
    self.filter.postMessage({ type: "init", meta });
    self.filter.addEventListener("message", (e) => e.data.type === "result" && self.onResult(e.data));

    self.workers = [];
    const ready = [];
    for (let i = 0; i < n; i++) {
      const w = new Worker(new URL("./gen-worker.js", import.meta.url), { type: "module" });
      ready.push(once(w, "ready"));
      w.postMessage({ type: "init", module, meta: slim, weights, kind, slots: self.slots, gemm: options.gemm });
      w.addEventListener("message", (e) => e.data.type === "cand" && self.onCandidate(w, e.data));
      self.workers.push(w);
    }
    const readies = await Promise.all([filterReady, ...ready]);
    self.info.workers = n;
    self.info.wide = readies.slice(1).filter((r) => r.wide).length;
    self.info.loadMs = performance.now() - t0;
    self.runId = 0;
    return self;
  }

  generate(overrides = {}, prompt = "", onUpdate = () => {}, seed = null) {
    const g = { ...this.meta.generation, ...overrides };
    const fail = (message) => Promise.resolve({ ok: false, text: "", message, accepted: 0, total: 0, candidates: [], stats: null });

    prompt = prompt || g.opening;
    if (prompt) {
      const reasons = this.safety.check(prompt, false);
      if (reasons.length) return fail(`Prompt rejected by the safety boundary: ${reasons.join(", ")}`);
    }
    const notes = [];
    const unknown = this.tok.unknownWords(prompt);
    if (unknown.length) {
      const shown = unknown.slice(0, 8).map((w) => `'${w}'`).join(", ");
      if (g.strict_prompt)
        return fail(`Prompt contains ${unknown.length} word(s) outside the model's vocabulary: ${shown}. Drop strict prompt mode to generate without those pieces.`);
      prompt = this.tok.sanitize(prompt);
      notes.push(`dropped ${unknown.length} out-of-vocabulary prompt word(s): ${shown}`);
    }
    const prefixIds = this.tok.encode(controlPrefix(g) + prompt, true);
    if (prefixIds.length >= this.meta.model.block_size) return fail("Prompt is longer than the model's context window.");

    this.cancel();
    const total = g.candidates;
    const run = this.run = {
      id: ++this.runId, g, prompt, notes, total, issued: 0, evaluated: 0, tokens: 0,
      t0: performance.now(), firstMs: null, best: null, candidates: [], onUpdate,
    };
    const tickets = this.workers.map(() => 0);
    while (run.issued < total) {
      const i = tickets.indexOf(Math.min(...tickets));
      if (tickets[i] >= this.slots) break;
      tickets[i]++; run.issued++;
    }
    this.workers.forEach((w, i) => {
      if (!tickets[i]) return;
      const s = seed === null ? crypto.getRandomValues(new Uint32Array(4)) : Uint32Array.of(seed, i + 1, 0x9e3779b9, 0x243f6a88);
      w.postMessage({ type: "run", runId: run.id, tickets: tickets[i], g, prefixIds, seed: s });
    });
    return new Promise((resolve) => { run.resolve = resolve; });
  }

  onCandidate(worker, { runId, ids }) {
    const run = this.run;
    if (!run || run.id !== runId) return;
    if (run.issued < run.total) { run.issued++; worker.postMessage({ type: "ticket", runId, n: 1 }); }
    this.filter.postMessage({ type: "eval", runId, ids, prompt: run.prompt, g: run.g });
  }

  onResult({ runId, c }) {
    const run = this.run;
    if (!run || run.id !== runId) return;
    run.evaluated++;
    run.tokens += c.tokens;
    run.candidates.push(c);
    if (c.accepted) {
      if (run.firstMs === null) run.firstMs = performance.now() - run.t0;
      if (!run.best || better(c, run.best)) run.best = c;
    }
    run.onUpdate(run);
    if (run.evaluated === run.total) this.finish(run);
  }

  finish(run, stopped = false) {
    if (this.run === run) this.run = null;
    const ms = performance.now() - run.t0;
    const accepted = run.candidates.filter((c) => c.accepted);
    const rejected = run.candidates.filter((c) => !c.accepted);
    const stats = {
      ms, firstMs: run.firstMs, tokens: run.tokens, tokPerSec: run.tokens / (ms / 1000),
      evaluated: run.evaluated, stopped, rejections: rejectionSummary(rejected),
    };
    const total = run.candidates.length;
    run.resolve(accepted.length
      ? { ok: true, text: run.best.text, best: run.best, accepted: accepted.length, total, candidates: run.candidates, stats,
          message: [`${accepted.length} of ${total} candidates passed.`, ...run.notes].join("; ") }
      : { ok: false, text: "", accepted: 0, total, candidates: run.candidates, stats,
          message: "No candidate passed the safety and copy filters this time. Press again." });
  }

  dispose() {
    this.cancel();
    for (const w of this.workers) w.terminate();
    this.filter.terminate();
  }

  cancel() {
    const run = this.run;
    if (!run) return;
    for (const w of this.workers) w.postMessage({ type: "cancel" });
    this.finish(run, true);
  }
}
