import { useState, useEffect } from 'react'
import './App.css'

const PLANES = ['sagittal', 'coronal', 'axial']
const NAMES = { abnormal: 'Abnormality', acl: 'ACL tear', meniscus: 'Meniscus tear' }
const name = l => NAMES[l] || l
const api = (p = '', o) => fetch('/api/v1/exams' + p, o).then(r => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
const when = (iso, d = 'medium') => new Date(iso.endsWith('Z') ? iso : iso + 'Z').toLocaleString([], { dateStyle: d, timeStyle: 'short' })
const pct = v => `${(v * 100).toFixed(1)}%`

/** Parse the .npy header client-side so the intake can confirm slices × H × W before upload. */
async function npyShape(file) {
  const head = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  if (String.fromCharCode(...head.slice(1, 6)) !== 'NUMPY') return null
  const v1 = head[6] === 1, start = v1 ? 10 : 12
  const len = v1 ? head[8] | (head[9] << 8) : head[8] | (head[9] << 8) | (head[10] << 16) | (head[11] << 24)
  const txt = new TextDecoder().decode(await file.slice(start, start + len).arrayBuffer())
  const m = txt.match(/'shape':\s*\(([^)]*)\)/)
  return m ? m[1].split(',').map(s => s.trim()).filter(Boolean).map(Number) : null
}

export default function App() {
  const [exams, setExams] = useState([])
  const [current, setCurrent] = useState(null) // exam id or null = new exam
  const refresh = () => api().then(setExams).catch(() => {})
  useEffect(() => { refresh() }, [])

  return (
    <div className="app">
      <aside className="side">
        <header className="brand">
          <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true"><rect width="28" height="28" rx="7" fill="var(--blue)" /><path d="M9 19.5V8.5h2.2v4.6l4.3-4.6h2.8l-4.5 4.7 4.8 6.3h-2.8l-3.5-4.7-1.1 1.1v3.6z" fill="#fff" /></svg>
          <div><strong>Knee MRI Detect</strong><span>Abnormality detection · Grad-CAM</span></div>
        </header>
        <button className="btn primary wide" onClick={() => { setCurrent(null); refresh() }} aria-pressed={current === null}>
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>New exam
        </button>
        <nav aria-label="Exam history" className="history">
          {exams.length === 0 && <p className="muted">Analysed exams will appear here.</p>}
          {exams.map(e => <ExamRow key={e.id} e={e} active={e.id === current} onClick={() => setCurrent(e.id)} />)}
        </nav>
        <p className="disclaimer">Research prototype. Decision support only — not a medical device.</p>
      </aside>
      <main className="main">
        {current === null
          ? <Intake onCreated={id => { setCurrent(id); refresh() }} />
          : <ExamView key={current} id={current} onChange={refresh} />}
      </main>
    </div>
  )
}

function ExamRow({ e, active, onClick }) {
  const pos = e.flagged
  return (
    <button className="row" aria-current={active || undefined} onClick={onClick}>
      <span className={`dot ${e.status}`} aria-label={e.status} />
      <span className="row-body">
        <span className="row-title">{e.patient_ref || `Exam #${e.id}`}</span>
        <span className="row-sub">#{e.id} · {e.planes.length}p · {when(e.created_at, 'short')}</span>
      </span>
      {e.status === 'done' && <span className={`count ${pos ? 'pos' : ''}`}>{pos ? `${pos} flagged` : 'clear'}</span>}
      {e.status === 'pending' && <span className="count">running</span>}
      {e.status === 'error' && <span className="count err">failed</span>}
    </button>
  )
}

function Intake({ onCreated }) {
  const [patient, setPatient] = useState('')
  const [files, setFiles] = useState({})       // plane -> {file, shape}
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const n = Object.keys(files).length

  async function pick(plane, file) {
    if (!file) return setFiles(f => { const { [plane]: _, ...rest } = f; return rest })
    setFiles(f => ({ ...f, [plane]: { file, shape: null } }))
    const shape = await npyShape(file)
    setFiles(f => (f[plane]?.file === file ? { ...f, [plane]: { file, shape } } : f))
  }

  async function submit(e) {
    e.preventDefault(); setBusy(true); setError('')
    const fd = new FormData(); fd.append('patient_ref', patient.trim())
    PLANES.forEach(p => files[p] && fd.append(p, files[p].file))
    try { onCreated((await api('', { method: 'POST', body: fd })).id) }
    catch (err) { setError(`Upload failed (${err.message}). Check the API is running on :8000.`); setBusy(false) }
  }

  return (
    <form className="intake" onSubmit={submit}>
      <h1>New exam</h1>
      <p className="lede">Upload one or more planes as MRNet-style <code>.npy</code> stacks (slices × H × W). The model ensembles whatever planes you provide.</p>
      <label className="field">
        <span>Patient reference <em>optional</em></span>
        <input value={patient} onChange={e => setPatient(e.target.value)} placeholder="e.g. MRN-0042" autoComplete="off" />
      </label>
      <div className="planes">
        {PLANES.map(p => {
          const f = files[p]
          return (
            <label key={p} className={`drop ${f ? 'filled' : ''}`}>
              <input type="file" accept=".npy" onChange={e => pick(p, e.target.files[0])} />
              <span className="drop-plane">{p}</span>
              {f ? (
                <>
                  <span className="drop-name" title={f.file.name}>{f.file.name}</span>
                  <span className="drop-meta">{f.shape ? `${f.shape[0]} slices · ${f.shape[1]}×${f.shape[2]}` : f.shape === null ? 'not a .npy stack' : 'reading…'}</span>
                </>
              ) : (
                <>
                  <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"><path d="M11 15V4m0 0L6.5 8.5M11 4l4.5 4.5M4 14v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>
                  <span className="drop-meta">Choose .npy</span>
                </>
              )}
            </label>
          )
        })}
      </div>
      {error && <p className="error" role="alert">{error}</p>}
      <div className="actions">
        <button className="btn primary" disabled={!n || busy}>{busy ? 'Uploading…' : n ? `Analyse ${n} plane${n === 1 ? '' : 's'}` : 'Analyse'}</button>
        <span className="muted">Inference runs in the background; you can scrub slices while it works.</span>
      </div>
    </form>
  )
}

function ExamView({ id, onChange }) {
  const [exam, setExam] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => {
    let stop = false, t
    const tick = () => api(`/${id}`).then(e => {
      if (stop) return
      setExam(prev => { if (prev?.status === 'pending' && e.status !== 'pending') onChange(); return e })
      if (e.status === 'pending') t = setTimeout(tick, 1500)
    }).catch(e => setErr(e.message))
    tick()
    return () => { stop = true; clearTimeout(t) }
  }, [id])

  if (err) return <p className="error" role="alert">Could not load exam #{id} ({err}).</p>
  if (!exam) return <div className="skeleton-page" aria-busy="true" />
  const planes = Object.keys(exam.planes)
  return (
    <article className="exam">
      <header className="exam-head">
        <div>
          <h1>{exam.patient_ref || `Exam #${exam.id}`}</h1>
          <p className="muted">#{exam.id} · {when(exam.created_at)} · {planes.map(p => `${p} ${exam.planes[p]}`).join(' · ')}</p>
        </div>
        <Status s={exam.status} />
      </header>
      <div className="grid">
        <Viewer exam={exam} planes={planes} />
        <Findings exam={exam} />
      </div>
    </article>
  )
}

function Status({ s }) {
  const txt = { pending: 'Analysing', done: 'Complete', error: 'Failed' }[s]
  return <span className={`status ${s}`}>{s === 'pending' && <i className="spin" aria-hidden="true" />}{txt}</span>
}

function Viewer({ exam, planes }) {
  const [plane, setPlane] = useState(planes[0])
  const [i, setI] = useState(Math.floor(exam.planes[planes[0]] / 2))
  const [heat, setHeat] = useState(null) // label whose Grad-CAM is shown
  const n = exam.planes[plane]
  const v = encodeURIComponent(exam.created_at)   // exam ids restart after a DB reset; keep cached slices per-exam
  const src = `/api/v1/exams/${exam.id}/slice/${plane}/${i}?v=${v}`
  const meta = exam.gradcam_meta || {}
  const heatLabels = Object.keys(exam.gradcam || {}).filter(l => meta.slices?.[l] != null)

  // Warm the cache for the current plane so scrubbing is instant.
  useEffect(() => { for (let k = 0; k < n; k++) new Image().src = `/api/v1/exams/${exam.id}/slice/${plane}/${k}?v=${v}` }, [exam.id, plane, n, v])

  // Findings → "show heatmap" events.
  useEffect(() => {
    const h = e => show(e.detail)
    window.addEventListener('show-heatmap', h); return () => window.removeEventListener('show-heatmap', h)
  }, [exam, meta])

  function show(l) { if (!l || !exam.gradcam[l] || meta.slices?.[l] == null) return setHeat(null); setHeat(l); setPlane(meta.plane); setI(meta.slices[l]) }
  function switchPlane(p) { setPlane(p); setI(Math.floor(exam.planes[p] / 2)); setHeat(null) }

  return (
    <section className="viewer card" aria-label="Slice viewer">
      <div className="toolbar">
        <fieldset className="seg" aria-label="Plane">
          {planes.map(p => (
            <label key={p}><input type="radio" name="plane" checked={plane === p} onChange={() => switchPlane(p)} /><span>{p}</span></label>
          ))}
        </fieldset>
        <div className="toolbar-right">
          {heatLabels.length > 0 && (
            <select className="heat" value={heat || ''} onChange={e => show(e.target.value)} aria-label="Grad-CAM heatmap">
              <option value="">Slices</option>
              {heatLabels.map(l => <option key={l} value={l}>Heatmap · {name(l)}</option>)}
            </select>
          )}
          <span className="counter"><b>{i + 1}</b> / {n}</span>
        </div>
      </div>
      <div className="frame">
        <img key={heat ? 'heat' + heat : src} src={heat ? exam.gradcam[heat] : src} alt={heat ? `${name(heat)} Grad-CAM heatmap on ${plane} slice ${i + 1}` : `${plane} slice ${i + 1} of ${n}`} draggable="false" />
        {heat && <span className="chip">Grad-CAM · {name(heat)} · slice {i + 1}<button onClick={() => setHeat(null)} aria-label="Hide heatmap">×</button></span>}
      </div>
      <input type="range" min="0" max={n - 1} value={i} onChange={e => { setI(+e.target.value); setHeat(null) }} aria-label={`${plane} slice`} aria-valuetext={`slice ${i + 1} of ${n}`} />
    </section>
  )
}

function Findings({ exam }) {
  const labels = Object.keys(exam.thresholds)
  if (exam.status === 'error') return (
    <section className="card findings"><h2>Findings</h2>
      <p className="error" role="alert">Inference failed: {exam.predictions?.error || 'unknown error'}. Check the stack shape is (slices, H, W) and model weights exist in MODEL_DIR.</p>
    </section>
  )
  const pending = exam.status === 'pending'
  const flagged = !pending && labels.filter(l => exam.predictions[l] >= exam.thresholds[l])
  return (
    <section className="card findings" aria-busy={pending}>
      <div className="findings-head">
        <h2>Findings</h2>
        {pending ? <span className="muted">Running the model across {Object.keys(exam.planes).length} plane{Object.keys(exam.planes).length === 1 ? '' : 's'}…</span>
          : <span className={`summary ${flagged.length ? 'pos' : 'neg'}`}>{flagged.length ? `${flagged.length} of ${labels.length} flagged` : 'No findings above threshold'}</span>}
      </div>
      {pending && <div className="indeterminate" aria-hidden="true" />}
      <ol className="list">
        {labels.map((l, k) => {
          const p = exam.predictions?.[l], t = exam.thresholds[l], pos = !pending && p >= t
          return (
            <li key={l} className={`finding ${pending ? 'pending' : pos ? 'pos' : 'neg'}`} style={{ '--i': k }}>
              <div className="f-head"><span className="f-name">{name(l)}</span>
                {!pending && <span className="f-verdict">{pos ? 'Positive' : 'Negative'}</span>}</div>
              <div className="bar" role="meter" aria-valuemin="0" aria-valuemax="100" aria-valuenow={pending ? undefined : Math.round(p * 100)} aria-label={`${name(l)} probability`}>
                <i style={{ '--p': pending ? 0 : p }} /><b style={{ '--t': t }} title={`threshold ${pct(t)}`} />
              </div>
              <div className="f-meta">
                <span className="num">{pending ? '—' : pct(p)}</span>
                <span className="muted">threshold {pct(t)}</span>
                {exam.gradcam?.[l] && exam.gradcam_meta?.slices?.[l] != null && <button className="link" onClick={() => window.dispatchEvent(new CustomEvent('show-heatmap', { detail: l }))}>Show heatmap</button>}
              </div>
            </li>
          )
        })}
      </ol>
      {exam.status === 'done' && (
        <footer className="findings-foot">
          <a className="btn" href={`/api/v1/exams/${exam.id}/report`} download>
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M7 1v8m0 0L3.5 5.5M7 9l3.5-3.5M1.5 10.5v1a1.5 1.5 0 0 0 1.5 1.5h8a1.5 1.5 0 0 0 1.5-1.5v-1" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>Download PDF report
          </a>
          <span className="muted">Probabilities are ensembled across uploaded planes; thresholds were tuned on the validation split.</span>
        </footer>
      )}
    </section>
  )
}
