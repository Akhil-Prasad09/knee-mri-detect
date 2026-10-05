// In-browser stand-in for the FastAPI backend, used by the GitHub Pages build (VITE_STATIC=1).
// It answers the same calls App.jsx makes (create, list, get, slice image, Grad-CAM) but runs the
// ONNX exports of the three plane models with ONNX Runtime Web. Exams live in memory only, so a
// scan never leaves the visitor's machine and is gone on reload.
//
// Preprocessing mirrors ml/data/transforms.py and the overlay mirrors ml/explain/gradcam.py;
// web/export.py checked the exported models against PyTorch.

const SIZE = 224, MEAN = 0.485, STD = 0.229, MAX_SLICES = 80
const BASE = import.meta.env.BASE_URL
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/'
const OPENCV = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.12.0-release.1/dist/opencv.js'

const exams = new Map()      // id -> exam record (same shape as GET /api/v1/exams/{id}) + private stacks
const sessions = {}
let nextId = 1, cfgP, ortP, cvP

const config = () => (cfgP ??= fetch(BASE + 'config.json').then(r => r.json()))
const loadOrt = () => (ortP ??= import(/* @vite-ignore */ ORT + 'ort.min.mjs').then(ort => { ort.env.wasm.wasmPaths = ORT; return ort }))
const loadCv = () => (cvP ??= new Promise((resolve, reject) => {
  const s = Object.assign(document.createElement('script'), { src: OPENCV, crossOrigin: 'anonymous', integrity: 'sha384-i8A4fJEsRcMFMyEEDNri/2MR12DhkFLlUF+9oxUrxs6prIcRj7YtWAQ1OJ+iE0C7' })
  s.onerror = () => reject(new Error('could not load OpenCV.js'))
  s.onload = async () => {
    let cv = window.cv   // the build exposes either a promise or a module with a callback
    if (cv instanceof Promise) cv = await cv
    else if (!cv.Mat) await new Promise(r => { cv.onRuntimeInitialized = r })
    resolve({ cv })   // wrapped: the Emscripten module has its own .then, so resolving with it loops forever
  }
  document.head.append(s)
}))

export async function api(path = '', opts = {}) {
  if (opts.method === 'POST') return create(opts.body)
  if (path === '') return [...exams.values()].reverse().map(summary)
  const e = exams.get(Number(path.slice(1)))
  if (!e) throw new Error('404 Not Found')
  return { ...e.public }
}

function summary({ public: e }) {
  const flagged = e.status === 'done' ? Object.keys(e.thresholds).filter(l => e.predictions[l] >= e.thresholds[l]).length : null
  return { id: e.id, patient_ref: e.patient_ref, status: e.status, created_at: e.created_at, planes: Object.keys(e.planes), flagged }
}

async function create(form) {
  const cfg = await config()
  const id = nextId++
  const stacks = {}
  for (const p of cfg.planes) {
    const f = form.get(p)
    if (f) stacks[p] = parseNpy(await f.arrayBuffer())
  }
  for (const [p, s] of Object.entries(stacks)) {
    if (s.shape.length !== 3) throw new Error(`${p}: expected (slices, H, W), got (${s.shape})`)
    if (s.shape[0] < 1 || s.shape[0] > MAX_SLICES) throw new Error(`${p}: expected 1 to ${MAX_SLICES} slices, got ${s.shape[0]}`)
  }
  const e = {
    stacks, slices: new Map(),
    public: {
      id, patient_ref: form.get('patient_ref') || '', status: 'pending', created_at: new Date().toISOString(),
      planes: Object.fromEntries(Object.entries(stacks).map(([p, s]) => [p, s.shape[0]])),
      predictions: {}, thresholds: cfg.thresholds, gradcam: {}, gradcam_meta: {}, progress: 'Loading the model runtime…',
    },
  }
  exams.set(id, e)
  infer(e, cfg)
  return { id, status: 'pending' }
}

async function infer(e, cfg) {
  const pub = e.public
  try {
    const [ort, { cv }] = await Promise.all([loadOrt(), loadCv()])
    const per = {}
    for (const plane of Object.keys(e.stacks)) {
      pub.progress = `Preprocessing the ${plane} stack…`
      const x = preprocessStack(cv, e.stacks[plane])
      if (!sessions[plane]) {
        pub.progress = `Downloading the ${plane} model (43 MB, cached after the first time)…`
        sessions[plane] = await session(ort, plane)
      }
      pub.progress = `Running the ${plane} model on ${e.stacks[plane].shape[0]} slices…`
      const S = e.stacks[plane].shape[0]
      const out = await sessions[plane].run({ x: new ort.Tensor('float32', x, [S, 3, SIZE, SIZE]) })
      per[plane] = { probs: Array.from(out.logits.data, v => 1 / (1 + Math.exp(-v))), best: Array.from(out.best.data, Number), cam: out.cam.data, x }
    }
    const planes = Object.keys(per)
    pub.predictions = Object.fromEntries(cfg.labels.map((l, i) => [l, planes.reduce((a, p) => a + per[p].probs[i], 0) / planes.length]))
    // Like the backend: Grad-CAM for each positive finding, on the first uploaded plane.
    const plane = planes[0], slices = {}
    cfg.labels.forEach((l, i) => {
      if (pub.predictions[l] < cfg.thresholds[l]) return
      slices[l] = per[plane].best[i]
      pub.gradcam[l] = camUrl(cv, per[plane], i)
    })
    if (Object.keys(slices).length) pub.gradcam_meta = { plane, slices }
    pub.status = 'done'
  } catch (err) {
    console.error(err)
    pub.status = 'error'; pub.predictions = { error: err.message }
  }
  delete pub.progress
}

async function session(ort, plane) {
  const eps = 'gpu' in navigator ? [['webgpu'], ['wasm']] : [['wasm']]
  let last
  for (const ep of eps) {
    try { return await ort.InferenceSession.create(`${BASE}models/${plane}.onnx`, { executionProviders: ep }) }
    catch (err) { last = err; console.warn(`${ep} failed`, err) }
  }
  throw last
}

/** Same as GET /slice/{plane}/{i}: min-max to 0..255, as a PNG data URL. */
export function sliceSrc(id, plane, i) {
  const e = exams.get(id)
  const key = `${plane}/${i}`
  if (!e || !e.stacks[plane]) return ''
  if (!e.slices.has(key)) {
    const { shape: [, H, W], data } = e.stacks[plane]
    const s = data.subarray(i * H * W, (i + 1) * H * W)
    let lo = Infinity, hi = -Infinity
    for (const v of s) { if (v < lo) lo = v; if (v > hi) hi = v }
    const c = Object.assign(document.createElement('canvas'), { width: W, height: H })
    const img = c.getContext('2d').createImageData(W, H)
    for (let k = 0; k < H * W; k++) {
      const g = hi > lo ? Math.trunc(((s[k] - lo) / (hi - lo)) * 255) : 0
      img.data.set([g, g, g, 255], k * 4)
    }
    c.getContext('2d').putImageData(img, 0, 0)
    e.slices.set(key, c.toDataURL())
  }
  return e.slices.get(key)
}

function parseNpy(buf) {
  const u8 = new Uint8Array(buf)
  if (u8[0] !== 0x93 || String.fromCharCode(...u8.slice(1, 6)) !== 'NUMPY') throw new Error('not a .npy file')
  const major = u8[6]
  const hlen = major === 1 ? u8[8] | (u8[9] << 8) : new DataView(buf).getUint32(8, true)
  const hstart = major === 1 ? 10 : 12
  const header = new TextDecoder().decode(u8.slice(hstart, hstart + hlen))
  const descr = header.match(/'descr':\s*'([^']+)'/)[1]
  if (/'fortran_order':\s*True/.test(header)) throw new Error('Fortran-ordered arrays are not supported')
  const shape = header.match(/'shape':\s*\(([^)]*)\)/)[1].split(',').map(s => s.trim()).filter(Boolean).map(Number)
  const types = { '|u1': Uint8Array, '<u1': Uint8Array, '<i2': Int16Array, '<u2': Uint16Array, '<f4': Float32Array, '<f8': Float64Array }
  const T = types[descr]
  if (!T) throw new Error(`unsupported dtype ${descr}`)
  const off = hstart + hlen, n = shape.reduce((a, b) => a * b, 1)
  return { shape, data: new T(buf.slice(off, off + n * T.BYTES_PER_ELEMENT)) }
}

// Resize to 224, min-max to uint8 (truncating like numpy), CLAHE, 3x3 Gaussian, ImageNet mean/std.
function preprocessSlice(cv, slice, H, W) {
  const src = cv.matFromArray(H, W, cv.CV_32F, Float32Array.from(slice))
  const r = new cv.Mat(), n = new cv.Mat(), c = new cv.Mat(), g = new cv.Mat()
  cv.resize(src, r, new cv.Size(SIZE, SIZE), 0, 0, cv.INTER_LINEAR)
  cv.normalize(r, n, 0, 255, cv.NORM_MINMAX)
  const m8 = cv.matFromArray(SIZE, SIZE, cv.CV_8U, Uint8Array.from(n.data32F, v => Math.trunc(v)))
  const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8))
  clahe.apply(m8, c)
  cv.GaussianBlur(c, g, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT)
  const out = Float32Array.from(g.data, v => (v / 255 - MEAN) / STD);
  [src, r, n, c, g, m8].forEach(m => m.delete()); clahe.delete()
  return out
}

function preprocessStack(cv, { shape: [S, H, W], data }) {
  const plane = SIZE * SIZE, x = new Float32Array(S * 3 * plane)
  for (let s = 0; s < S; s++) {
    const p = preprocessSlice(cv, data.subarray(s * H * W, (s + 1) * H * W), H, W)
    for (let ch = 0; ch < 3; ch++) x.set(p, (s * 3 + ch) * plane)   // grey repeated to 3 channels
  }
  return x
}

// Overlay like pytorch_grad_cam: scale CAM to [0,1], bilinear resize, JET colours, 50/50 blend, renormalise.
function camUrl(cv, r, i) {
  const s = r.best[i], plane = SIZE * SIZE
  const base = r.x.subarray(s * 3 * plane, s * 3 * plane + plane)
  let bmin = Infinity, bmax = -Infinity
  for (const v of base) { if (v < bmin) bmin = v; if (v > bmax) bmax = v }
  const cam = Float32Array.from(r.cam.subarray(i * 49, (i + 1) * 49))
  const cmin = Math.min(...cam)
  for (let k = 0; k < 49; k++) cam[k] -= cmin
  const cmax = Math.max(...cam)
  for (let k = 0; k < 49; k++) cam[k] /= 1e-7 + cmax
  const small = cv.matFromArray(7, 7, cv.CV_32F, cam), big = new cv.Mat()
  cv.resize(small, big, new cv.Size(SIZE, SIZE), 0, 0, cv.INTER_LINEAR)
  const mask = big.data32F, blend = new Float32Array(plane * 3)
  let mx = 0
  for (let k = 0; k < plane; k++) {
    const g = (base[k] - bmin) / (bmax - bmin + 1e-6)
    jet(Math.trunc(255 * mask[k])).forEach((v, c) => { const o = 0.5 * v + 0.5 * g; blend[k * 3 + c] = o; if (o > mx) mx = o })
  }
  small.delete(); big.delete()
  const canvas = Object.assign(document.createElement('canvas'), { width: SIZE, height: SIZE })
  const img = canvas.getContext('2d').createImageData(SIZE, SIZE)
  for (let k = 0; k < plane; k++) {
    for (let c = 0; c < 3; c++) img.data[k * 4 + c] = Math.trunc((255 * blend[k * 3 + c]) / mx)
    img.data[k * 4 + 3] = 255
  }
  canvas.getContext('2d').putImageData(img, 0, 0)
  return canvas.toDataURL()
}

// Close piecewise-linear match to OpenCV's COLORMAP_JET (RGB in [0,1]).
function jet(v) {
  const x = v / 255, f = t => Math.min(1, Math.max(0, t))
  return [f(1.5 - Math.abs(4 * x - 3)), f(1.5 - Math.abs(4 * x - 2)), f(1.5 - Math.abs(4 * x - 1))]
}
