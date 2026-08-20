import { useState, useEffect } from 'react'

const PLANES = ['sagittal', 'coronal', 'axial']

export default function App() {
  const [files, setFiles] = useState({})
  const [patient, setPatient] = useState('')
  const [exam, setExam] = useState(null)

  async function submit(e) {
    e.preventDefault()
    const fd = new FormData()
    fd.append('patient_ref', patient)
    PLANES.forEach(p => files[p] && fd.append(p, files[p]))
    const r = await fetch('/api/v1/exams', { method: 'POST', body: fd })
    setExam(await r.json())
  }

  useEffect(() => {
    if (!exam || exam.status !== 'pending') return
    const t = setInterval(async () => setExam(await (await fetch(`/api/v1/exams/${exam.id}`)).json()), 2000)
    return () => clearInterval(t)
  }, [exam])

  return (
    <main style={{ maxWidth: 760, margin: '2rem auto', fontFamily: 'system-ui' }}>
      <h1>Knee MRI Abnormality Detection</h1>
      <p style={{ color: '#666' }}>Research prototype — decision support only, not a diagnosis.</p>
      <form onSubmit={submit}>
        <input placeholder="Patient reference" value={patient} onChange={e => setPatient(e.target.value)} />
        {PLANES.map(p => (
          <label key={p} style={{ display: 'block', margin: '8px 0' }}>{p}: <input type="file" accept=".npy" onChange={e => setFiles({ ...files, [p]: e.target.files[0] })} /></label>
        ))}
        <button>Analyse</button>
      </form>
      {exam && (
        <section>
          <h2>Exam #{exam.id} — {exam.status}</h2>
          {exam.status === 'done' && (
            <>
              <table border="1" cellPadding="6"><tbody>
                {Object.entries(exam.predictions).map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{(v * 100).toFixed(1)}%</td><td style={{ color: v >= exam.thresholds[k] ? 'crimson' : 'green' }}>{v >= exam.thresholds[k] ? 'POSITIVE' : 'negative'}</td></tr>
                ))}
              </tbody></table>
              {Object.entries(exam.gradcam).map(([k, url]) => <figure key={k}><img src={url} alt={k} width="256" /><figcaption>Grad-CAM: {k}</figcaption></figure>)}
              <a href={`/api/v1/exams/${exam.id}/report`}>Download PDF report</a>
            </>
          )}
          {exam.status === 'error' && <pre>{JSON.stringify(exam.predictions)}</pre>}
        </section>
      )}
    </main>
  )
}
