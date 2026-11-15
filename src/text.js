
const W = "\\p{L}\\p{N}_";
const WS = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const BOUNDARY = `(?:(?<=[${W}])(?![${W}])|(?<![${W}])(?=[${W}]))`;

const WORD_CHAR = /[\p{L}\p{N}_]/u;

function boundary(pattern, at) {
  const next = pattern[at + 2];
  if (next !== undefined && WORD_CHAR.test(next)) return `(?<![${W}])`;
  let before = pattern.slice(0, at);
  if (before.endsWith("\\w*")) before = before.slice(0, -3);
  else if (before.endsWith("\\w+") || before.endsWith("\\w")) return `(?![${W}])`;
  const prev = before[before.length - 1];
  if (prev !== undefined && WORD_CHAR.test(prev) && before[before.length - 2] !== "\\") return `(?![${W}])`;
  return BOUNDARY;
}

export function pyRegex(pattern, flags = "") {
  const src = pattern.replace(/\\([bwsWS])/g, (_, c, at) =>
    c === "b" ? boundary(pattern, at) : c === "w" ? `[${W}]` : c === "s" ? `[${WS}]` : c === "W" ? `[^${W}]` : `[^${WS}]`);
  return new RegExp(src, flags.includes("u") ? flags : flags + "u");
}

const PIECE_SRC = `[${W}]+|[^${W}${WS}]|[${WS}]+`;
const PIECE_G = new RegExp(PIECE_SRC, "gu");
const PIECE_Y = new RegExp(PIECE_SRC, "yu");
const SPACE_1 = new RegExp(`^[${WS}]`, "u");
const STRIP = new RegExp(`^[${WS}]+|[${WS}]+$`, "gu");
const LSTRIP = new RegExp(`^[${WS}]+`, "u");
const ALPHA = /\p{L}/u;
const UPPER = /\p{Lu}/u;

export const pieces = (text) => text.match(PIECE_G) || [];
export const isSpace = (piece) => SPACE_1.test(piece);
export const strip = (text) => text.replace(STRIP, "");
export const lstrip = (text) => text.replace(LSTRIP, "");
export const isAlpha = (ch) => ALPHA.test(ch);
export const isUpper = (ch) => UPPER.test(ch);
export const wordTokens = (text) => pieces(text).filter((p) => !isSpace(p));

export function pyLen(text) {
  let n = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c < 0xdc00 || c > 0xdfff) n++;
  }
  return n;
}

export function count(text, needle) {
  let n = 0, i = 0;
  while ((i = text.indexOf(needle, i)) !== -1) { n++; i += needle.length; }
  return n;
}

export class Tokenizer {
  constructor(meta) {
    this.tokens = meta.vocab;
    this.stoi = new Map(meta.vocab.map((t, i) => [t, i]));
    this.ids = meta.ids;
    this.specials = new Set(meta.specials);
    this.atomic = meta.vocab.filter((t) => t.startsWith("<") && t.endsWith(">")).sort((a, b) => b.length - a.length);
  }

  encode(text, addBos = false) {
    const out = addBos ? [this.ids.bos] : [];
    let i = 0;
    while (i < text.length) {
      if (text[i] === "<") {
        const tok = this.atomic.find((t) => text.startsWith(t, i));
        if (tok !== undefined) { out.push(this.stoi.get(tok)); i += tok.length; continue; }
      }
      PIECE_Y.lastIndex = i;
      const m = PIECE_Y.exec(text);
      const piece = m ? m[0] : String.fromCodePoint(text.codePointAt(i));
      out.push(this.stoi.get(piece) ?? this.ids.unk);
      i += piece.length;
    }
    return out;
  }

  decode(ids) {
    let s = "";
    for (const id of ids) {
      if (id === this.ids.eos || id === this.ids.pad) break;
      const t = this.tokens[id] ?? "<UNK>";
      if (!this.specials.has(t)) s += t;
    }
    return s;
  }

  unknownWords(prompt) {
    const unknown = [];
    for (const p of pieces(prompt))
      if (!isSpace(p) && !this.stoi.has(p) && !unknown.includes(p)) unknown.push(p);
    return unknown;
  }

  sanitize(prompt) {
    const unknown = new Set(this.unknownWords(prompt));
    return unknown.size ? pieces(prompt).filter((p) => !unknown.has(p)).join("") : prompt;
  }
}

export const controlPrefix = (g) =>
  `<${g.length.toUpperCase()}><CONTEMPT_${g.intensity}><FORM_${g.form.toUpperCase()}><TARGET_${g.target.toUpperCase()}>`;
