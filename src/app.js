import { Generator } from "./generator.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const num = (k) => (params.has(k) ? Number(params.get(k)) : undefined);

const generateButton = $("generate");
const randomButton = $("random");
const post = $("post");
const status = $("status");
let gen;

function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}

for (const tab of document.querySelectorAll('[role="tab"]'))
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll('[role="tab"]')) {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      $(t.getAttribute("aria-controls")).hidden = !on;
    }
    generateButton.hidden = tab.id !== "tab-generate";
    randomButton.hidden = tab.id !== "tab-universe";
  });

async function generate() {
  if (!gen || generateButton.disabled) return;
  generateButton.disabled = true;
  setStatus("Generating...");
  try {
    const result = await gen.generate(
      { target: "pear", length: "medium", intensity: 3 },
      "",
      undefined,
      num("seed") ?? null,
    );
    if (result.ok) {
      post.textContent = result.text;
      setStatus("");
    } else {
      setStatus("No post generated. Try again.", true);
    }
  } catch (err) {
    console.error(err);
    setStatus("Could not generate a post. Try again.", true);
  } finally {
    generateButton.disabled = false;
  }
}

generateButton.addEventListener("click", generate);

let current = null;
randomButton.addEventListener("click", () => {
  const universe = gen.meta.universe;
  const posts = current && universe.length > 1 ? universe.filter((entry) => entry.id !== current.id) : universe;
  current = posts[Math.floor(Math.random() * posts.length)];
  $("u-post").textContent = current.text.trim();
});

try {
  gen = await Generator.load(
    { workers: num("workers"), slots: num("slots"), kernel: params.get("kernel"), gemm: num("gemm") },
  );
  generateButton.disabled = false;
  randomButton.disabled = gen.meta.universe.length === 0;
  setStatus("");
  $("foot").textContent = `${gen.meta.parameters.toLocaleString("en-US")} parameters`;
} catch (err) {
  console.error(err);
  setStatus(`Could not start: ${err.message}`, true);
}
