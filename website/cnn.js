// Pure JavaScript CNN inference for the Traffic Sign Recognition model.
// Reads model_config.json + weights.bin exported from Keras. No libraries needed.
(function (root) {
  const EPS_DEFAULT = 1e-3;

  async function loadModel(base) {
    const [cfg, buf, names] = await Promise.all([
      fetch(base + 'model_config.json').then(r => r.json()),
      fetch(base + 'weights.bin').then(r => r.arrayBuffer()),
      fetch(base + 'class_names.json').then(r => r.json()),
    ]);
    return build(cfg, new Float32Array(buf), names);
  }

  function build(cfg, flat, names) {
    const layers = cfg.layers.map(l => ({
      ...l,
      w: l.weights.map(w => {
        const size = w.shape.reduce((a, b) => a * b, 1);
        return { shape: w.shape, data: flat.subarray(w.offset, w.offset + size) };
      }),
    }));
    return { layers, names, input: cfg.input_shape };
  }

  // x: Float32Array [H*W*C]
  function conv3x3(x, H, W, C, kernel, bias, relu) {
    const O = kernel.shape[3];
    const k = kernel.data, out = new Float32Array(H * W * O);
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < W; xx++) {
        const o0 = (y * W + xx) * O;
        for (let o = 0; o < O; o++) out[o0 + o] = bias.data[o];
        for (let ky = 0; ky < 3; ky++) {
          const iy = y + ky - 1;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = xx + kx - 1;
            if (ix < 0 || ix >= W) continue;
            const i0 = (iy * W + ix) * C;
            const k0 = (ky * 3 + kx) * C * O;
            for (let c = 0; c < C; c++) {
              const v = x[i0 + c];
              if (v === 0) continue;
              const kk = k0 + c * O;
              for (let o = 0; o < O; o++) out[o0 + o] += v * k[kk + o];
            }
          }
        }
        if (relu) for (let o = 0; o < O; o++) if (out[o0 + o] < 0) out[o0 + o] = 0;
      }
    }
    return out;
  }

  function batchNorm(x, C, gamma, beta, mean, variance, eps) {
    const scale = new Float32Array(C), shift = new Float32Array(C);
    for (let c = 0; c < C; c++) {
      scale[c] = gamma.data[c] / Math.sqrt(variance.data[c] + eps);
      shift[c] = beta.data[c] - mean.data[c] * scale[c];
    }
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) {
      const c = i % C;
      out[i] = x[i] * scale[c] + shift[c];
    }
    return out;
  }

  function maxPool2(x, H, W, C) {
    const h = H >> 1, w = W >> 1, out = new Float32Array(h * w * C);
    for (let y = 0; y < h; y++) for (let xx = 0; xx < w; xx++) for (let c = 0; c < C; c++) {
      const a = x[((2 * y) * W + 2 * xx) * C + c];
      const b = x[((2 * y) * W + 2 * xx + 1) * C + c];
      const d = x[((2 * y + 1) * W + 2 * xx) * C + c];
      const e = x[((2 * y + 1) * W + 2 * xx + 1) * C + c];
      out[(y * w + xx) * C + c] = Math.max(a, b, d, e);
    }
    return out;
  }

  function dense(x, kernel, bias, act) {
    const [I, O] = kernel.shape, k = kernel.data, out = new Float32Array(O);
    for (let o = 0; o < O; o++) out[o] = bias.data[o];
    for (let i = 0; i < I; i++) {
      const v = x[i];
      if (v === 0) continue;
      const row = i * O;
      for (let o = 0; o < O; o++) out[o] += v * k[row + o];
    }
    if (act === 'relu') for (let o = 0; o < O; o++) if (out[o] < 0) out[o] = 0;
    if (act === 'softmax') {
      let m = -Infinity; for (let o = 0; o < O; o++) m = Math.max(m, out[o]);
      let s = 0; for (let o = 0; o < O; o++) { out[o] = Math.exp(out[o] - m); s += out[o]; }
      for (let o = 0; o < O; o++) out[o] /= s;
    }
    return out;
  }

  // input: Float32Array of 32*32*3 values scaled to 0..1 (RGB, row-major)
  function predictProbs(model, input) {
    let [H, W, C] = model.input;
    let x = input;
    for (const l of model.layers) {
      switch (l.type) {
        case 'Conv2D':
          x = conv3x3(x, H, W, C, l.w[0], l.w[1], l.activation === 'relu');
          C = l.w[0].shape[3];
          break;
        case 'BatchNormalization':
          x = batchNorm(x, C, l.w[0], l.w[1], l.w[2], l.w[3], l.epsilon || EPS_DEFAULT);
          break;
        case 'MaxPooling2D':
          x = maxPool2(x, H, W, C); H >>= 1; W >>= 1;
          break;
        case 'Dense':
          x = dense(x, l.w[0], l.w[1], l.activation);
          break;
        default: break; // Dropout, Flatten: nothing to do at inference
      }
    }
    return x;
  }

  function topK(probs, names, k) {
    return Array.from(probs).map((p, i) => ({ i, p })).sort((a, b) => b.p - a.p).slice(0, k)
      .map(r => ({ id: r.i, name: names[r.i], prob: r.p }));
  }

  const api = { loadModel, build, predictProbs, topK };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TrafficSignCNN = api;
})(typeof window !== 'undefined' ? window : globalThis);
