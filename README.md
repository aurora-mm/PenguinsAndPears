# Penguins and Pears

A small Transformer, written from scratch, that generates Russian-language internet rants about pears. It runs entirely in the browser tab; nothing is sent anywhere.

The forward pass, attention, sampling and a small batched generation scheduler live in a WebAssembly module; the loader, workers and UI around it are plain JavaScript. Weights ship as a single f16 blob and are unpacked on load.

Parameters: 2,403,840.

## How it runs

- One WASM module backs all generation. Several slots run in parallel inside it; Web Workers drive them from JS.
- On load the engine calibrates its GEMM kernel (wide vs. narrow) by timing both and keeping the faster.
- Sampling supports temperature, top-k, top-p and repetition penalties.
- The "Universe" tab serves a random rewrite from the training set, with no inference involved.

## Running locally

The page uses WASM, Web Workers and `fetch`, so opening `index.html` over `file://` will not work. Serve the directory with any static server, e.g.:

```
python -m http.server 8000
```

then open `http://localhost:8000/`.

## License

MIT. Thanks to the unnamed Russian edgelord who inspired the whole thing.
