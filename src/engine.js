
export function pickKernel(plainBytes, relaxedBytes) {
  if (relaxedBytes && WebAssembly.validate(relaxedBytes)) return { bytes: relaxedBytes, relaxed: true };
  return { bytes: plainBytes, relaxed: false };
}

export class Engine {
  constructor(module, meta, slots) {
    this.meta = meta;
    this.slots = slots;
    this.x = new WebAssembly.Instance(module, {}).exports;
    const m = meta.model;
    this.V = meta.vocab.length;
    if (this.x.init(this.V, m.d_model, m.n_heads, m.n_layers, m.d_ff, m.block_size, slots) !== 0)
      throw new Error("kernel rejected the model shape");
    const mem = () => new Uint8Array(this.x.memory.buffer);
    const forb = this.x.forbidden_ptr(), skip = this.x.skip_ptr();
    for (const i of meta.forbidden) mem()[forb + i] = 1;
    for (const i of meta.filler) mem()[skip + i] = 1;
  }

  calibrate() {
    const time = (wide) => {
      this.x.set_gemm(wide);
      let best = Infinity;
      for (let i = 0; i < 4; i++) {
        const t0 = performance.now();
        this.x.calibrate(40);
        best = Math.min(best, performance.now() - t0);
      }
      return best;
    };
    time(1);
    this.wide = time(1) < time(0);
    this.x.set_gemm(this.wide ? 1 : 0);
    return this.wide;
  }

  loadWeights(buffer, kind) {
    const n = this.x.weights_len();
    if (kind === "f16") {
      if (buffer.byteLength !== n * 2) throw new Error("weights size mismatch");
      const lo = new Uint8Array(buffer, 0, n), hi = new Uint8Array(buffer, n, n);
      const out = new Float32Array(this.x.memory.buffer, this.x.weights_ptr(), n);
      const table = halfTable();
      for (let i = 0; i < n; i++) out[i] = table[lo[i] | (hi[i] << 8)];
    } else {
      if (buffer.byteLength !== n * 4) throw new Error("weights size mismatch");
      new Float32Array(this.x.memory.buffer, this.x.weights_ptr(), n).set(new Float32Array(buffer));
    }
  }

  configure(g) {
    this.x.configure(g.temperature, g.top_k, g.top_p, g.repetition_penalty, g.immediate_repeat_penalty,
      g.max_new_tokens, this.meta.ids.eos);
  }

  seed(words) {
    if (!words) words = crypto.getRandomValues(new Uint32Array(4));
    this.x.seed(words[0], words[1], words[2], words[3]);
  }

  setPrefix(ids) {
    if (ids.length > this.meta.model.block_size) throw new Error("prompt longer than the context window");
    new Int32Array(this.x.memory.buffer, this.x.prefix_ptr(), ids.length).set(ids);
    this.prefixLen = ids.length;
    this.x.set_prefix(ids.length);
  }

  start(slot) { return this.x.start(slot); }
  step() { return this.x.step(); }
  active() { return this.x.active(); }
  state(slot) { return this.x.slot_state(slot); }
  release(slot) { this.x.release(slot); }

  tokens(slot) {
    const n = this.x.slot_len(slot);
    return Array.from(new Int32Array(this.x.memory.buffer, this.x.slot_ids(slot), n).subarray(this.prefixLen));
  }

  logits(row) {
    return new Float32Array(this.x.memory.buffer, this.x.logits_ptr() + row * this.V * 4, this.V);
  }
  prefixLogits() {
    return new Float32Array(this.x.memory.buffer, this.x.prefix_logits_ptr(), this.V);
  }
}

let HALF = null;
function halfTable() {
  if (HALF) return HALF;
  HALF = new Float32Array(65536);
  const bits = new Uint32Array(HALF.buffer);
  for (let h = 0; h < 65536; h++) {
    const s = h >>> 15, e = (h >>> 10) & 31, m = h & 1023;
    if (e === 0) HALF[h] = (s ? -1 : 1) * m * 2 ** -24;
    else bits[h] = ((s << 31) | ((e === 31 ? 255 : e + 112) << 23) | (m << 13)) >>> 0;
  }
  return HALF;
}

export async function gunzipIfNeeded(buffer) {
  const bytes = new Uint8Array(buffer);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return buffer;
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export async function unpackData(buffer) {
  const bytes = new Uint8Array(await gunzipIfNeeded(buffer));
  const len = new DataView(bytes.buffer, bytes.byteOffset).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + len)));
  const off = (4 + len + 3) & ~3;
  meta.memStream = new Uint16Array(bytes.buffer.slice(bytes.byteOffset + off, bytes.byteOffset + off + meta.mem_stream_len * 2));
  return meta;
}
