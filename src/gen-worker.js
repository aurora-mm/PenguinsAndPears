
import { Engine } from "./engine.js";

let engine = null, run = null, pumping = false;

const channel = new MessageChannel();
let wake = null;
channel.port1.onmessage = () => { const w = wake; wake = null; w && w(); };
const tick = () => new Promise((r) => { wake = r; channel.port2.postMessage(0); });

function emitFinished() {
  for (let s = 0; s < engine.slots; s++)
    if (engine.state(s) === 2) {
      postMessage({ type: "cand", runId: run.id, ids: engine.tokens(s) });
      engine.release(s);
    }
}

async function pump() {
  if (pumping) return;
  pumping = true;
  while (run) {
    let started = false;
    for (let s = 0; s < engine.slots && run.tickets > 0; s++)
      if (engine.state(s) === 0) { run.tickets--; engine.start(s); started = true; }
    if (started) emitFinished();  
    if (!engine.active()) {
      if (run.tickets > 0) continue;
      break;
    }
    const t0 = performance.now();
    let fin = 0;
    do { fin = engine.step(); } while (!fin && performance.now() - t0 < 6);
    if (fin) emitFinished();
    await tick();  
  }
  pumping = false;
}

onmessage = ({ data }) => {
  if (data.type === "init") {
    engine = new Engine(data.module, data.meta, data.slots);
    engine.loadWeights(data.weights, data.kind);
    const wide = data.gemm === undefined ? engine.calibrate() : (engine.x.set_gemm(data.gemm), !!data.gemm);
    postMessage({ type: "ready", wide });
  } else if (data.type === "run") {
    for (let s = 0; s < engine.slots; s++) engine.release(s);
    run = { id: data.runId, tickets: data.tickets };
    engine.configure(data.g);
    engine.seed(data.seed);
    engine.setPrefix(data.prefixIds);
    pump();
  } else if (data.type === "ticket") {
    if (run && run.id === data.runId) { run.tickets += data.n; pump(); }
  } else if (data.type === "cancel") {
    run = null;
  }
};
