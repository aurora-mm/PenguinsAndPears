
import { Tokenizer } from "./text.js";
import { Judge } from "./filters.js";

let tok = null, judge = null;

onmessage = ({ data }) => {
  if (data.type === "init") {
    tok = new Tokenizer(data.meta);
    judge = new Judge(data.meta);
    postMessage({ type: "ready" });
  } else if (data.type === "eval") {
    const c = judge.evaluate(tok.decode(data.ids), data.prompt, data.g);
    c.tokens = data.ids.length;
    postMessage({ type: "result", runId: data.runId, c });
  }
};
