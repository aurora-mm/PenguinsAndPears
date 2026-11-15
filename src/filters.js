
import { pyRegex, pyLen, count, strip, lstrip, isAlpha, isUpper, wordTokens } from "./text.js";


export class SafetyFilter {
  constructor(rules) {
    const compile = (list) => pyRegex(list.map((p) => `(?:${p})`).join("|"), "iu");
    this.groups = [
      [compile(rules.identity), "identity-category reference"],
      [compile(rules.dehumanization), "dehumanizing construction"],
      [compile(rules.threat), "threat/wish-of-harm construction"],
      [compile(rules.person_abuse), "direct personal abuse"],
    ];
    this.safeTerms = rules.safe_terms;
  }

  check(text, requireSafeTarget = true) {
    const reasons = [];
    for (const [pattern, reason] of this.groups)
      if (pattern.test(text)) reasons.push(reason);
    if (requireSafeTarget) {
      const low = text.toLowerCase();
      if (!this.safeTerms.some((t) => low.includes(t))) reasons.push("no explicit safe target anchor");
    }
    return reasons;
  }
}


const NUMBERED = pyRegex("(?:^|\\n)\\s*[123][.)]");
const RUN = /(.)\1{4,}/u;

export function styleScore(text, style) {
  if (!strip(text)) return -10.0;
  const low = text.toLowerCase();
  let score = 0;
  score += Math.min(count(text, "?"), 2) * 0.5;
  score += Math.min(count(text, "!"), 3) * 0.25;
  score += Math.min(count(text, "—"), 3) * 0.15;
  for (const p of style.pivots) if (low.includes(p)) score += 0.15;
  for (const p of style.understatements) if (low.includes(p)) score += 0.25;
  if (NUMBERED.test(text)) score += 0.7;

  let cased = 0, upper = 0;
  for (const ch of text) if (isAlpha(ch)) { cased++; if (isUpper(ch)) upper++; }
  const ratio = upper / Math.max(1, cased);
  if (ratio >= 0.03 && ratio <= 0.20) score += 0.5;
  else if (ratio > 0.35) score -= 0.8;

  score += Math.exp(-(((pyLen(text) - 240) / 180) ** 2));
  if (RUN.test(low)) score -= 2.0;
  return score;
}

export function startsLikeASentence(text) {
  const s = lstrip(text);
  if (!s) return false;
  const first = String.fromCodePoint(s.codePointAt(0));
  return isAlpha(first) ? isUpper(first) : true;
}


class RefMatcher {
  constructor(b) {
    this.b = b;
    const n = b.length;
    const lists = new Map();
    for (let j = 0; j < n; j++) {
      const l = lists.get(b[j]);
      if (l) l.push(j); else lists.set(b[j], [j]);
    }
    this.count = new Map();
    this.b2j = new Map();
    const ntest = Math.floor(n / 100) + 1;
    for (const [c, l] of lists) {
      this.count.set(c, l.length);
      if (!(n >= 200 && l.length > ntest)) this.b2j.set(c, Int32Array.from(l));
    }
    this.len = [new Int32Array(n + 1), new Int32Array(n + 1)];
    this.stamp = [new Int32Array(n + 1), new Int32Array(n + 1)];
    this.clock = 1;
  }

  quickRatio(aCount, la) {
    let m = 0;
    for (const [c, na] of aCount) {
      const nb = this.count.get(c);
      if (nb) m += Math.min(na, nb);
    }
    const total = la + this.b.length;
    return total ? (2.0 * m) / total : 1.0;
  }

  ratio(a) {
    const b = this.b, b2j = this.b2j, la = a.length, lb = b.length;
    const queue = [0, la, 0, lb];
    let matches = 0;
    while (queue.length) {
      const bhi = queue.pop(), blo = queue.pop(), ahi = queue.pop(), alo = queue.pop();
      let besti = alo, bestj = blo, bestsize = 0;
      this.clock += 2;
      for (let i = alo; i < ahi; i++) {
        const row = ++this.clock, cur = row & 1, prev = cur ^ 1;
        const js = b2j.get(a[i]);
        if (!js) continue;
        const lc = this.len[cur], sc = this.stamp[cur], lp = this.len[prev], sp = this.stamp[prev];
        for (let x = 0; x < js.length; x++) {
          const j = js[x];
          if (j < blo) continue;
          if (j >= bhi) break;
          const k = (sp[j] === row - 1 ? lp[j] : 0) + 1;  
          lc[j + 1] = k; sc[j + 1] = row;
          if (k > bestsize) { besti = i - k + 1; bestj = j - k + 1; bestsize = k; }
        }
      }
      while (besti > alo && bestj > blo && a[besti - 1] === b[bestj - 1]) { besti--; bestj--; bestsize++; }
      while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] === b[bestj + bestsize]) bestsize++;
      if (bestsize) {
        matches += bestsize;
        if (alo < besti && blo < bestj) queue.push(alo, besti, blo, bestj);
        if (besti + bestsize < ahi && bestj + bestsize < bhi) queue.push(besti + bestsize, ahi, bestj + bestsize, bhi);
      }
    }
    const total = la + lb;
    return total ? (2.0 * matches) / total : 1.0;
  }
}

const codePoints = (text) => {
  const out = [];
  for (const ch of text) out.push(ch.codePointAt(0));
  return out;
};

export class SimilarityIndex {
  constructor(references) {
    this.matchers = references.map((r) => new RefMatcher(codePoints(r)));
  }

  maxSimilarity(text) {
    const a = codePoints(text), la = a.length;
    const aCount = new Map();
    for (const c of a) aCount.set(c, (aCount.get(c) || 0) + 1);
    let best = 0.0;
    for (const m of this.matchers) {
      const lb = m.b.length, total = la + lb;
      const realQuick = total ? (2.0 * Math.min(la, lb)) / total : 1.0;
      if (realQuick <= best) continue;
      if (m.quickRatio(aCount, la) <= best) continue;
      const r = m.ratio(a);
      if (r > best) best = r;
    }
    return best;
  }
}


export class MemorizationIndex {
  constructor(words, stream, n = 8, seedN = 4) {
    this.n = n; this.seedN = seedN;
    this.wordId = new Map(words.map((w, i) => [w, i + 1]));
    this.stream = stream;
    const len = stream.length;
    let bits = 16;
    while ((1 << bits) < len * 2) bits++;
    this.mask = (1 << bits) - 1;
    this.head = new Int32Array(1 << bits).fill(-1);
    this.next = new Int32Array(len).fill(-1);
    let sep = -1;  
    for (let i = 0; i < seedN - 1 && i < len; i++) if (stream[i] === 0) sep = i;
    for (let i = 0; i + seedN <= len; i++) {
      if (stream[i + seedN - 1] === 0) sep = i + seedN - 1;
      if (sep >= i) continue;
      const h = this.hash(stream, i);
      this.next[i] = this.head[h];
      this.head[h] = i;
    }
    this.sets = new Map();
  }

  hash(arr, i) {
    let h = arr[i];
    for (let k = 1; k < this.seedN; k++) h = (Math.imul(h, 0x9e3779b1) + arr[i + k]) | 0;
    h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
    return (h ^ (h >>> 13)) & this.mask;
  }

  ids(tokens) {
    const out = new Int32Array(tokens.length);
    for (let i = 0; i < tokens.length; i++) out[i] = this.wordId.get(tokens[i]) ?? -1;
    return out;
  }

  run(t, i, p) {
    const s = this.stream, lt = t.length, ls = s.length;
    let k = 0;
    while (i + k < lt && p + k < ls && t[i + k] === s[p + k]) k++;
    return k;
  }

  longestExactSpan(t) {
    const seedN = this.seedN;
    let best = 0;
    for (let i = 0; i + seedN <= t.length; i++) {
      if (t.length - i <= best) break;  
      for (let p = this.head[this.hash(t, i)]; p >= 0; p = this.next[p]) {
        const k = this.run(t, i, p);
        if (k >= seedN && k > best) best = k;
      }
    }
    if (best === 0 && t.length) {
      for (let width = Math.min(seedN - 1, t.length); width >= 1; width--)
        if (this.containsAny(t, width)) { best = width; break; }
    }
    return best;
  }

  containsAny(t, width) {
    let set = this.sets.get(width);
    if (!set) {
      set = new Set();
      const s = this.stream;
      outer: for (let i = 0; i + width <= s.length; i++) {
        let key = 0;
        for (let k = 0; k < width; k++) { if (s[i + k] === 0) continue outer; key = key * 65536 + s[i + k]; }
        set.add(key);
      }
      this.sets.set(width, set);
    }
    outer: for (let i = 0; i + width <= t.length; i++) {
      let key = 0;
      for (let k = 0; k < width; k++) { if (t[i + k] <= 0) continue outer; key = key * 65536 + t[i + k]; }
      if (set.has(key)) return true;
    }
    return false;
  }

  ngramOverlap(t) {
    const n = this.n;
    if (t.length < n) return 0.0;
    const total = t.length - n + 1;
    let hit = 0;
    for (let i = 0; i < total; i++)
      for (let p = this.head[this.hash(t, i)]; p >= 0; p = this.next[p])
        if (this.run(t, i, p) >= n) { hit++; break; }
    return hit / total;
  }

  report(text) {
    const t = this.ids(wordTokens(text));
    const span = this.longestExactSpan(t);
    return { span, spanRatio: t.length ? span / t.length : 0.0, ngramOverlap: this.ngramOverlap(t), words: t.length };
  }
}


const round = (x, d) => Math.round(x * 10 ** d) / 10 ** d;

export class Judge {
  constructor(meta) {
    this.safety = new SafetyFilter(meta.safety);
    this.similarity = new SimilarityIndex(meta.references);
    this.memorization = new MemorizationIndex(meta.mem_words, meta.memStream);
    this.style = meta.style;
  }

  evaluate(generated, prompt, g) {
    const text = strip(prompt + strip(generated));
    const reasons = this.safety.check(text, true);
    const similarity = this.similarity.maxSimilarity(text);
    if (similarity > g.max_copy_similarity) reasons.push(`too similar to corpus (${similarity.toFixed(3)})`);
    if (pyLen(text) < g.min_length) reasons.push("too short");
    if (g.require_sentence_start && !prompt && !startsLikeASentence(text)) reasons.push("does not start a sentence");
    const rep = this.memorization.report(text);
    if (g.max_exact_span && rep.span > g.max_exact_span) reasons.push(`verbatim training span (${rep.span} tokens)`);
    if (g.max_span_ratio && rep.spanRatio > g.max_span_ratio)
      reasons.push(`near-total reconstruction (${Math.round(rep.spanRatio * 100)}% of the post)`);
    return {
      text,
      styleScore: round(styleScore(text, this.style), 3),
      similarity: round(similarity, 3),
      span: rep.span,
      ngramOverlap: round(rep.ngramOverlap, 4),
      reasons,
      accepted: reasons.length === 0,
    };
  }
}

export const better = (a, b) =>
  a.styleScore !== b.styleScore ? a.styleScore > b.styleScore : a.similarity < b.similarity;

export function rejectionSummary(rejected) {
  const counts = new Map();
  for (const c of rejected)
    for (const r of c.reasons) {
      const key = r.split(" (")[0];
      counts.set(key, (counts.get(key) || 0) + 1);
    }
  return [...counts].sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} (${n})`).join("; ");
}
