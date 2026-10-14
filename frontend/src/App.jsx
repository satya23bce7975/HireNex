import React from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleGauge,
  Cpu,
  Database,
  Download,
  FileCheck2,
  FileText,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  Search,
  ServerCog,
  Sparkles,
  Target,
  Trash2,
  Trophy,
  UploadCloud,
  Zap,
} from 'lucide-react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const API = import.meta.env.VITE_API_URL || ''

const api = async (path, options = {}) => {
  const response = await fetch(`${API}${path}`, options)
  const type = response.headers.get('content-type') || ''
  const data = type.includes('application/json') ? await response.json() : await response.text()
  if (!response.ok) {
    const message = typeof data === 'object' ? data.detail?.message || data.detail || data.message : data
    throw new Error(message || 'Request failed')
  }
  return data
}

function Spinner({ small = false }) {
  return <span className={`spinner ${small ? 'spinner-small' : ''}`} aria-hidden="true" />
}

function Pill({ children, tone = 'neutral' }) {
  return <span className={`pill pill-${tone}`}>{children}</span>
}

function MetricCard({ icon: Icon, label, value, caption, tone = 'cyan' }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon tone-${tone}`}><Icon size={18} /></div>
      <div className="metric-copy">
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        {caption && <div className="metric-caption">{caption}</div>}
      </div>
    </div>
  )
}

function SectionTitle({ step, icon: Icon, title, text, badge }) {
  return (
    <div className="section-title-row">
      <div className="section-title-left">
        <div className="section-number">{step}</div>
        <div className="section-icon"><Icon size={18} /></div>
        <div>
          <div className="section-title">{title}</div>
          <div className="section-text">{text}</div>
        </div>
      </div>
      {badge && badge}
    </div>
  )
}

function PipelineStep({ icon: Icon, title, subtitle, state = 'idle' }) {
  return (
    <div className={`pipeline-step pipeline-${state}`}>
      <div className="pipeline-node">
        {state === 'running' ? <Spinner small /> : state === 'success' ? <CheckCircle2 size={17} /> : <Icon size={17} />}
      </div>
      <div className="pipeline-copy">
        <strong>{title}</strong>
        <span>{subtitle}</span>
      </div>
    </div>
  )
}

function EmptyState({ icon: Icon, title, text }) {
  return (
    <div className="empty-state">
      <div className="empty-icon"><Icon size={22} /></div>
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  )
}

function JobProgress({ job }) {
  if (!job || typeof job !== 'object') return null
  const terminal = ['completed', 'failed'].includes(job.status)
  const calculated = job.total ? Math.round((Number(job.completed || 0) / Number(job.total)) * 100) : 0
  const suppliedPercentage = Number(job.percentage)
  const percentage = job.status === 'completed' ? 100 : (Number.isFinite(suppliedPercentage) && suppliedPercentage > 0 ? suppliedPercentage : calculated)
  const hasResumeErrors = job.status === 'completed' && Number(job.failed || 0) > 0
  const StateIcon = job.status === 'completed' && !hasResumeErrors ? CheckCircle2 : ['failed'].includes(job.status) || hasResumeErrors ? AlertTriangle : null
  const stateClass = hasResumeErrors ? 'job-partial' : `job-${job.status || 'running'}`
  return (
    <div className={`job-box ${stateClass} job-${job.kind || 'task'}`} role="status" aria-live="polite">
      <div className="job-head">
        <div className="job-title-line">
          {StateIcon ? <StateIcon size={15} /> : <Spinner small />}
          <strong>{job.message || (job.status === 'queued' ? 'Queued…' : 'Processing…')}</strong>
        </div>
        <span>{job.total ? `${job.completed || 0}/${job.total}` : (terminal ? 'Done' : 'Working…')}</span>
      </div>
      <div className="progress-track"><div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }} /></div>
      <div className="job-meta-line">
        <span>{job.phase ? job.phase.replaceAll('_', ' ') : 'processing'}</span>
        <strong>{job.total ? `${Math.max(0, Math.min(100, percentage))}%` : (terminal ? '100%' : '…')}</strong>
      </div>
      {job.current && <div className="job-current">Current file: <strong>{job.current}</strong></div>}
      {job.status === 'failed' && job.error && <div className="job-error">{job.error}</div>}
    </div>
  )
}

function Workspace({ status, refreshStatus, showNotice, setShowNotice, activeJob, setActiveJob }) {
  const [files, setFiles] = React.useState([])
  const [dragging, setDragging] = React.useState(false)
  const [uploading, setUploading] = React.useState(false)
  const [removeConfirm, setRemoveConfirm] = React.useState(false)
  const [parserJobs, setParserJobs] = React.useState({ nlp: null, llm: null })
  const [query, setQuery] = React.useState('')
  const [searchBusy, setSearchBusy] = React.useState(false)
  const [searchResult, setSearchResult] = React.useState(null)
  const fileRef = React.useRef(null)
  const folderRef = React.useRef(null)
  const zipRef = React.useRef(null)

  const conversion = status?.conversion || { total: 0, converted: 0, failed: 0, pending: 0, not_converted: 0, state: 'empty', next_application: 'APP000001' }
  const candidates = searchResult?.candidates || []

  const addFiles = (incoming) => {
    const next = Array.from(incoming || []).filter((file) => {
      const name = file.name.toLowerCase()
      return name.endsWith('.pdf') || name.endsWith('.zip')
    })
    setFiles((current) => {
      const seen = new Set(current.map((f) => `${f.name}:${f.size}`))
      return [...current, ...next.filter((f) => !seen.has(`${f.name}:${f.size}`))]
    })
  }

  const uploadAndConvert = async () => {
    if (!files.length) return
    setUploading(true)
    setShowNotice(null)
    const selectedCount = files.length
    setActiveJob(null)
    try {
      const form = new FormData()
      files.forEach((file) => form.append('files', file))
      const result = await api('/api/upload', { method: 'POST', body: form })
      setFiles([])
      await refreshStatus()
      if (result.job_id) {
        setActiveJob({
          job_id: result.job_id,
          kind: 'markdown_conversion',
          status: 'queued',
          phase: 'queued',
          total: result.created_count || selectedCount,
          completed: 0,
          failed: 0,
          percentage: 0,
          message: 'Queued PDF → Markdown conversion…',
        })
      }
      setShowNotice({ type: 'success', text: `${result.created_count} new resume(s) accepted. PDF → Markdown conversion started.` })
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setUploading(false)
    }
  }

  const clearDataset = async () => {
    if (!removeConfirm) return
    try {
      await api('/api/data/clear', { method: 'POST' })
      setFiles([])
      setActiveJob(null)
      setParserJobs({ nlp: null, llm: null })
      setRemoveConfirm(false)
      setSearchResult(null)
      setShowNotice({ type: 'success', text: 'Current dataset and generated outputs were removed. New uploads start from APP000001.' })
      await refreshStatus()
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    }
  }

  const waitForJob = React.useCallback(async (jobId, kind) => {
    let finished = false
    let finalJob = null
    while (!finished) {
      const job = await api(`/api/jobs/${jobId}`)
      finalJob = job
      setParserJobs((current) => ({ ...current, [kind]: job }))
      finished = ['completed', 'failed'].includes(job.status)
      if (!finished) await new Promise((resolve) => setTimeout(resolve, 700))
    }
    await refreshStatus()
    return finalJob
  }, [refreshStatus])

  const processParser = async (kind) => {
    try {
      setShowNotice(null)
      const response = await api(`/api/process/${kind}`, { method: 'POST' })
      setParserJobs((current) => ({
        ...current,
        [kind]: { job_id: response.job_id, kind, status: 'queued', phase: 'queued', message: `Queued ${kind.toUpperCase()} parser…`, completed: 0, total: 0, failed: 0, percentage: 0 },
      }))
      const finalJob = await waitForJob(response.job_id, kind)
      if (finalJob?.status === 'failed') {
        setShowNotice({ type: 'error', text: finalJob.error || `${kind.toUpperCase()} processing failed.` })
      } else {
        const failures = Number(finalJob?.failed || 0)
        setShowNotice({ type: failures ? 'error' : 'success', text: failures ? `${kind.toUpperCase()} finished with ${failures} resume error(s). Check the output JSON for details.` : `${kind.toUpperCase()} processing completed.` })
      }
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    }
  }

  const searchCandidates = async () => {
    if (!query.trim()) return
    try {
      setSearchBusy(true)
      const result = await api('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      setSearchResult(result)
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setSearchBusy(false)
    }
  }

  const nlpState = parserJobs.nlp?.status === 'running' ? 'running' : status?.nlp_records ? 'success' : 'idle'
  const llmState = parserJobs.llm?.status === 'running' ? 'running' : status?.llm_records ? 'success' : 'idle'
  const conversionRunning = activeJob?.kind === 'markdown_conversion' && ['queued', 'running'].includes(activeJob.status)
  const conversionState = conversionRunning ? 'running' : conversion.state === 'success' ? 'success' : 'idle'

  return (
    <div className="page-stack">
      <section className="hero-panel">
        <div className="hero-grid">
          <div>
            <Pill tone="cyan"><Sparkles size={13} /> AI RECRUITMENT WORKSPACE</Pill>
            <h1>Turn resumes into<br /><span>recruitment intelligence.</span></h1>
            <p>Upload resumes, convert them into canonical Markdown, extract structured candidate profiles, and search them using hybrid retrieval — all in one responsive workspace.</p>
            <div className="hero-actions">
              <a href="#upload" className="hero-button"><UploadCloud size={17} /> Start screening <ArrowRight size={16} /></a>
              <span className="hero-note"><Zap size={14} /> React UI stays responsive during backend processing</span>
            </div>
          </div>
          <div className="hero-orbit">
            <div className="orbit-ring ring-a" />
            <div className="orbit-ring ring-b" />
            <div className="hero-core"><BrainCircuit size={42} /><span>NLP • LLM • RAG</span></div>
            <div className="orbit-chip chip-one">PDF</div>
            <div className="orbit-chip chip-two">NLP</div>
            <div className="orbit-chip chip-three">RAG</div>
          </div>
        </div>
      </section>

      <section className="metric-strip">
        <MetricCard icon={FileText} label="PDF resumes" value={conversion.total} caption={`${conversion.failed} failed • ${conversion.pending} pending`} tone="cyan" />
        <MetricCard icon={FileCheck2} label="Markdown" value={`${conversion.converted}/${conversion.total || 0}`} caption={conversion.total ? `${conversion.converted / conversion.total * 100 >= 0 ? Math.round((conversion.converted / conversion.total) * 100) : 0}% converted` : 'Waiting for data'} tone="violet" />
        <MetricCard icon={BrainCircuit} label="NLP records" value={status?.nlp_records || 0} caption="Traditional extraction" tone="lime" />
        <MetricCard icon={Sparkles} label="LLM records" value={status?.llm_records || 0} caption="Gemini structured extraction" tone="magenta" />
        <MetricCard icon={Target} label="Next application" value={conversion.next_application} caption="Auto-assigned by HireNex" tone="amber" />
      </section>

      <section id="upload" className="panel panel-glow">
        <SectionTitle step="01" icon={UploadCloud} title="Upload & Convert" text="Drop PDF resumes, a ZIP, or an entire folder." badge={<Pill tone={conversion.state === 'success' ? 'success' : conversion.state === 'incomplete' ? 'warning' : 'neutral'}>{conversion.state === 'success' ? 'Ready' : conversion.state === 'incomplete' ? 'Needs attention' : 'Waiting'}</Pill>} />

        <div className={`drop-zone ${dragging ? 'is-dragging' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files) }}>
          <div className="drop-icon"><UploadCloud size={30} /></div>
          <h3>Drag & drop your resume dataset</h3>
          <p>PDF files • ZIP archives • multiple resumes • entire folder</p>
          <div className="drop-actions">
            <button type="button" className="button button-secondary" onClick={() => fileRef.current?.click()}><FileText size={16} /> Choose PDFs</button>
            <button type="button" className="button button-secondary button-zip" onClick={() => zipRef.current?.click()}><Layers3 size={16} /> Choose ZIP</button>
            <button type="button" className="button button-secondary" onClick={() => folderRef.current?.click()}><Database size={16} /> Choose folder</button>
          </div>
          <input ref={fileRef} hidden type="file" accept="application/pdf,.pdf" multiple onChange={(e) => addFiles(e.target.files)} />
          <input ref={zipRef} hidden type="file" accept="application/zip,application/x-zip-compressed,.zip" multiple onChange={(e) => addFiles(e.target.files)} />
          <input ref={folderRef} hidden type="file" accept="application/pdf,.pdf" multiple webkitdirectory="true" directory="true" onChange={(e) => addFiles(e.target.files)} />
        </div>

        {files.length > 0 && (
          <div className="selected-files">
            <div className="selected-head"><strong>{files.length} file(s) selected</strong><button className="ghost-button" onClick={() => setFiles([])}>Clear selection</button></div>
            <div className="selected-list">{files.slice(0, 6).map((file) => <div className="selected-row" key={`${file.name}:${file.size}`}><span><span className="file-kind">{file.name.toLowerCase().endsWith('.zip') ? 'ZIP' : 'PDF'}</span>{file.name}</span><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>)}{files.length > 6 && <div className="selected-more">+ {files.length - 6} more files</div>}</div>
          </div>
        )}

        {showNotice && <div className={`alert ${showNotice.type === 'error' ? 'alert-error' : 'alert-success'}`}>{showNotice.type === 'error' ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}{showNotice.text}</div>}

        <div className="action-row">
          <button className="button button-primary" disabled={uploading || !files.length} onClick={uploadAndConvert}>{uploading ? <><Spinner small /> Uploading…</> : <><Play size={16} /> Upload & convert</>}</button>
          <label className="confirm-box"><input type="checkbox" checked={removeConfirm} onChange={(e) => setRemoveConfirm(e.target.checked)} /> I understand this removes the current dataset</label>
          <button className="button button-danger" disabled={!removeConfirm} onClick={clearDataset}><Trash2 size={16} /> Remove uploaded data</button>
        </div>

        {activeJob && <JobProgress job={activeJob} />}

        <div className="pipeline">
          <PipelineStep icon={FileText} title="PDF" subtitle="Source resumes" state={conversion.total ? 'success' : 'idle'} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={FileCheck2} title="Markdown" subtitle="Canonical text" state={conversionState} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={BrainCircuit} title="NLP" subtitle="Rule + spaCy" state={nlpState} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={Sparkles} title="LLM" subtitle="Structured extraction" state={llmState} />
          <ChevronRight className="pipeline-arrow" size={16} />
          <PipelineStep icon={Search} title="RAG" subtitle="Recruiter search" state={searchResult ? 'success' : 'idle'} />
        </div>

        <div className="dataset-grid">
          <MetricCard icon={FileText} label="PDF resumes" value={conversion.total} caption="Original source files" tone="cyan" />
          <MetricCard icon={FileCheck2} label="Converted" value={conversion.converted} caption="Markdown available" tone="violet" />
          <MetricCard icon={AlertTriangle} label="Failed" value={conversion.failed} caption="Needs retry" tone="rose" />
          <MetricCard icon={RefreshCw} label="Pending" value={conversion.pending} caption="Not yet processed" tone="amber" />
        </div>

        {conversion.failed_rows?.length > 0 && (
          <details className="error-drawer">
            <summary>View {conversion.failed_rows.length} conversion failure(s)</summary>
            <div className="table-scroll small-table"><table><thead><tr><th>Application</th><th>File</th><th>Error</th></tr></thead><tbody>{conversion.failed_rows.map((row) => <tr key={row.Application_No}><td>{row.Application_No}</td><td>{row.Source_File}</td><td>{row.Error}</td></tr>)}</tbody></table></div>
          </details>
        )}
      </section>

      <section className="two-column">
        <div className="panel">
          <SectionTitle step="02" icon={BrainCircuit} title="Traditional NLP" text="Fast deterministic extraction using rules + spaCy." badge={<Pill tone={status?.nlp_records ? 'success' : 'neutral'}>{status?.nlp_records ? `${status.nlp_records} records` : 'Not run'}</Pill>} />
          <div className="engine-card engine-nlp"><div className="engine-visual"><Gauge size={25} /><span>STRUCTURED</span></div><div><strong>Rule-based candidate extraction</strong><p>Names, contacts, education, skills, projects, experience, certifications, achievements.</p></div></div>
          <button className="button button-primary full" disabled={parserJobs.nlp?.status === 'running' || !conversion.converted} onClick={() => processParser('nlp')}>{parserJobs.nlp?.status === 'running' ? <><Spinner small /> Processing NLP…</> : <><BrainCircuit size={16} /> Process NLP</>}</button>
          {parserJobs.nlp && <JobProgress job={parserJobs.nlp} />}
          {parserJobs.nlp?.status === 'failed' && <div className="mini-error">{parserJobs.nlp.error}</div>}
          <div className="download-grid"><a className={`download-chip ${!status?.nlp_records ? 'disabled' : ''}`} href={`${API}/api/download/nlp/json`}><Download size={14} /> JSON</a><a className={`download-chip ${!status?.nlp_excel ? 'disabled' : ''}`} href={`${API}/api/download/nlp/xlsx`}><Download size={14} /> Excel</a></div>
        </div>

        <div className="panel">
          <SectionTitle step="03" icon={Sparkles} title="LLM Parsing" text="Semantic extraction with structured Gemini output." badge={<Pill tone={status?.llm_records ? 'success' : 'neutral'}>{status?.llm_records ? `${status.llm_records} records` : 'Not run'}</Pill>} />
          <div className="engine-card engine-llm"><div className="engine-visual"><Sparkles size={25} /><span>SEMANTIC</span></div><div><strong>Context-aware resume parsing</strong><p>Flexible interpretation for skills, projects, education, experience and profile details.</p></div></div>
          <button className="button button-primary full" disabled={parserJobs.llm?.status === 'running' || !conversion.converted} onClick={() => processParser('llm')}>{parserJobs.llm?.status === 'running' ? <><Spinner small /> Processing LLM…</> : <><Sparkles size={16} /> Process LLM</>}</button>
          {parserJobs.llm && <JobProgress job={parserJobs.llm} />}
          {parserJobs.llm?.status === 'failed' && <div className="mini-error">{parserJobs.llm.error}</div>}
          <div className="download-grid"><a className={`download-chip ${!status?.llm_records ? 'disabled' : ''}`} href={`${API}/api/download/llm/json`}><Download size={14} /> JSON</a><a className={`download-chip ${!status?.llm_excel ? 'disabled' : ''}`} href={`${API}/api/download/llm/xlsx`}><Download size={14} /> Excel</a></div>
        </div>
      </section>

      <section className="panel search-panel">
        <SectionTitle step="04" icon={Search} title="Recruiter Search" text="Ask in natural language and retrieve the most relevant candidates." badge={<Pill tone="cyan"><Zap size={12} /> Hybrid RAG</Pill>} />
        <div className="example-chips">
          {[
            'Show candidates with at least 2 years of experience and Python',
            'Find ECE candidates with FPGA experience',
            'Find Mechanical candidates with SolidWorks',
          ].map((example) => <button key={example} onClick={() => setQuery(example)}>{example}</button>)}
        </div>
        <div className="search-line"><div className="search-input-wrap"><Search size={17} /><input value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && searchCandidates()} placeholder="Ask a recruiter question…" /></div><button className="button button-primary" disabled={searchBusy || !query.trim()} onClick={searchCandidates}>{searchBusy ? <><Spinner small /> Searching…</> : <><Search size={16} /> Search candidates</>}</button></div>

        {searchBusy && <div className="search-loading"><Spinner /><div><strong>Understanding query → filtering → semantic ranking</strong><span>The React interface remains interactive while the backend runs.</span></div></div>}

        {searchResult && (
          <div className="results-panel">
            <div className="result-header"><div><Pill tone="success"><CheckCircle2 size={12} /> Search complete</Pill><h3>{candidates.length} candidates found</h3><p>{searchResult.answer || 'Candidates ranked using HireNex hybrid retrieval.'}</p></div>{searchResult.excel_url && <a className="button button-secondary" href={`${API}${searchResult.excel_url}`} download><Download size={15} /> Download Excel</a>}</div>
            {candidates.length ? <div className="table-scroll"><table><thead><tr><th>Application</th><th>Name</th><th>Experience</th><th>Skills</th><th>CGPA</th><th>College</th></tr></thead><tbody>{candidates.map((candidate) => <tr key={candidate.application_no}><td><span className="app-id">{candidate.application_no}</span></td><td><strong>{candidate.name || '—'}</strong></td><td>{candidate.total_experience_years ?? '—'} yrs</td><td><div className="skill-wrap">{(candidate.skills || []).slice(0, 6).map((skill) => <span className="skill-chip" key={skill}>{skill}</span>)}</div></td><td>{candidate.max_cgpa ?? '—'}</td><td>{candidate.college || '—'}</td></tr>)}</tbody></table></div> : <EmptyState icon={Search} title="No matching candidates" text="Try relaxing one of the structured filters or ask a broader semantic question." />}
            {searchResult.filters && <details className="query-details"><summary>View query interpretation</summary><pre>{JSON.stringify(searchResult.filters, null, 2)}</pre></details>}
          </div>
        )}
      </section>
    </div>
  )
}

function Evaluation({ status, activeJob, setActiveJob, showNotice, setShowNotice, refreshStatus }) {
  const [summary, setSummary] = React.useState(null)
  const [loading, setLoading] = React.useState(true)
  const [running, setRunning] = React.useState(false)
  const [evalJob, setEvalJob] = React.useState(null)

  const loadSummary = React.useCallback(async () => {
    try {
      const data = await api('/api/evaluation/summary')
      setSummary(data.result || null)
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setLoading(false)
    }
  }, [setShowNotice])

  React.useEffect(() => { loadSummary() }, [loadSummary])

  const runEvaluation = async () => {
    try {
      setRunning(true)
      setShowNotice(null)
      const { job_id } = await api('/api/evaluation/run', { method: 'POST' })
      setActiveJob({ job_id, kind: 'evaluation', status: 'queued', phase: 'queued', message: 'Queued benchmark evaluation…', completed: 0, total: 0, failed: 0, percentage: 0 })
      let finished = false
      let finalJob = null
      while (!finished) {
        const job = await api(`/api/jobs/${job_id}`)
        finalJob = job
        setEvalJob(job)
        finished = ['completed', 'failed'].includes(job.status)
        if (!finished) await new Promise((resolve) => setTimeout(resolve, 1000))
      }
      await loadSummary()
      await refreshStatus()
      setShowNotice({ type: finalJob?.status === 'failed' ? 'error' : 'success', text: finalJob?.status === 'failed' ? (finalJob.error || 'Evaluation failed.') : 'Evaluation run completed.' })
    } catch (error) {
      setShowNotice({ type: 'error', text: error.message })
    } finally {
      setRunning(false)
      setActiveJob(null)
    }
  }

  const parserRows = summary ? Object.values(summary.parsers || {}) : []
  const retrieval = summary?.retrieval?.aggregate
  const conversion = summary?.conversion
  const parserChart = parserRows.map((row) => ({ name: row.parser === 'Traditional NLP' ? 'NLP' : 'LLM', F1: +(row.overall_f1_percent || 0).toFixed(1), Accuracy: +(row.scalar_accuracy_percent || 0).toFixed(1), Skills: +((row.skills?.f1 || 0) * 100).toFixed(1) }))
  const retrievalChart = retrieval ? [
    { name: 'Precision@10', value: +(retrieval.precision_at_10 * 100).toFixed(1) },
    { name: 'Recall@10', value: +(retrieval.recall_at_10 * 100).toFixed(1) },
    { name: 'MRR', value: +(retrieval.mrr * 100).toFixed(1) },
    { name: 'nDCG@10', value: +(retrieval.ndcg_at_10 * 100).toFixed(1) },
  ] : []

  return (
    <div className="page-stack">
      <section className="hero-panel evaluation-hero">
        <div>
          <Pill tone="violet"><BarChart3 size={13} /> EVALUATION CENTER</Pill>
          <h1>Make the system <span>measurable.</span></h1>
          <p>Evaluate conversion, traditional NLP, LLM extraction, and recruiter retrieval against the HireNex benchmark dataset.</p>
        </div>
        <div className="evaluation-hero-icon"><Trophy size={52} /></div>
      </section>

      <section className="metric-strip">
        <MetricCard icon={Database} label="Benchmark records" value={summary?.benchmark?.ground_truth_records ?? '—'} caption="Ground-truth resumes" tone="cyan" />
        <MetricCard icon={Search} label="Benchmark queries" value={summary?.benchmark?.recruiter_queries ?? '—'} caption="Recruiter test cases" tone="violet" />
        <MetricCard icon={FileCheck2} label="Conversion success" value={conversion ? `${conversion.success_rate_percent}%` : '—'} caption="PDF → Markdown" tone="lime" />
        <MetricCard icon={BrainCircuit} label="NLP F1" value={summary?.parsers?.nlp ? `${summary.parsers.nlp.overall_f1_percent}%` : '—'} caption="Field extraction" tone="amber" />
        <MetricCard icon={Sparkles} label="LLM F1" value={summary?.parsers?.llm ? `${summary.parsers.llm.overall_f1_percent}%` : '—'} caption="Field extraction" tone="magenta" />
      </section>

      <section className="panel evaluation-toolbar">
        <div>
          <div className="toolbar-title"><CircleGauge size={18} /> Evaluation status</div>
          <p>{loading ? 'Checking benchmark availability…' : summary ? `Latest run: ${summary.evaluated_at}` : 'No evaluation run yet.'}</p>
        </div>
        <div className="toolbar-actions">
          {status?.evaluation?.benchmark_available ? <Pill tone="success"><CheckCircle2 size={12} /> Benchmark loaded</Pill> : <Pill tone="warning"><AlertTriangle size={12} /> Benchmark missing</Pill>}
          <button className="button button-primary" disabled={running || !status?.evaluation?.benchmark_available} onClick={runEvaluation}>{running ? <><Spinner small /> Running benchmark…</> : <><Play size={16} /> Run full evaluation</>}</button>
        </div>
      </section>

      {running && <JobProgress job={evalJob || { message: 'Running evaluation benchmark…', completed: 0, total: 0 }} />}

      {!summary ? (
        <section className="panel"><EmptyState icon={BarChart3} title="No evaluation results yet" text="Run the benchmark after generating NLP/LLM outputs. The results will appear here with charts and per-query metrics." /></section>
      ) : (
        <>
          <section className="two-column">
            <div className="panel chart-panel"><SectionTitle step="A" icon={BarChart3} title="Parser quality" text="Compare extraction quality across the two parsing approaches." /><div className="chart-box"><ResponsiveContainer width="100%" height={320}><BarChart data={parserChart} margin={{ top: 10, right: 15, left: -15, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" stroke="#26314a" /><XAxis dataKey="name" stroke="#8fa0bc" /><YAxis domain={[0, 100]} stroke="#8fa0bc" /><Tooltip contentStyle={{ background: '#101a2e', border: '1px solid #2d3b58', borderRadius: 12, color: '#eef4ff' }} /><Legend /><Bar dataKey="F1" fill="#22d3ee" radius={[7, 7, 0, 0]} /><Bar dataKey="Accuracy" fill="#a78bfa" radius={[7, 7, 0, 0]} /><Bar dataKey="Skills" fill="#84cc16" radius={[7, 7, 0, 0]} /></BarChart></ResponsiveContainer></div></div>
            <div className="panel chart-panel"><SectionTitle step="B" icon={Target} title="Retrieval quality" text="Top-k ranking quality over the recruiter benchmark." />{retrieval?.available && retrievalChart.length ? <div className="chart-box"><ResponsiveContainer width="100%" height={320}><BarChart data={retrievalChart} layout="vertical" margin={{ top: 10, right: 15, left: 25, bottom: 5 }}><CartesianGrid strokeDasharray="3 3" stroke="#26314a" /><XAxis type="number" domain={[0, 100]} stroke="#8fa0bc" /><YAxis type="category" dataKey="name" width={85} stroke="#8fa0bc" /><Tooltip contentStyle={{ background: '#101a2e', border: '1px solid #2d3b58', borderRadius: 12, color: '#eef4ff' }} /><Bar dataKey="value" fill="#f472b6" radius={[0, 8, 8, 0]} /></BarChart></ResponsiveContainer></div> : <EmptyState icon={Target} title="Retrieval benchmark unavailable" text="Run the full evaluation with a configured retrieval backend and API key." />}</div>
          </section>

          <section className="metric-strip evaluation-kpis">
            <MetricCard icon={Target} label="Precision@10" value={retrieval?.available ? `${(retrieval.precision_at_10 * 100).toFixed(1)}%` : '—'} caption={retrieval?.available ? 'Top-10 relevance' : 'Awaiting retrieval config'} tone="cyan" />
            <MetricCard icon={Search} label="Recall@10" value={retrieval?.available ? `${(retrieval.recall_at_10 * 100).toFixed(1)}%` : '—'} caption={retrieval?.available ? 'Relevant candidates found' : 'Awaiting retrieval config'} tone="violet" />
            <MetricCard icon={Trophy} label="MRR" value={retrieval?.available ? retrieval.mrr.toFixed(3) : '—'} caption={retrieval?.available ? 'First relevant rank' : 'Awaiting retrieval config'} tone="lime" />
            <MetricCard icon={CircleGauge} label="nDCG@10" value={retrieval?.available ? `${(retrieval.ndcg_at_10 * 100).toFixed(1)}%` : '—'} caption={retrieval?.available ? 'Ranking quality' : 'Awaiting retrieval config'} tone="magenta" />
            <MetricCard icon={Gauge} label="Avg query latency" value={retrieval?.available ? `${retrieval.avg_latency_seconds.toFixed(2)}s` : '—'} caption={retrieval?.available ? 'End-to-end benchmark' : 'Awaiting retrieval config'} tone="amber" />
          </section>

          <section className="panel">
            <SectionTitle step="C" icon={FileCheck2} title="Field-level extraction metrics" text="Useful evidence for your capstone evaluation chapter." />
            <div className="table-scroll"><table><thead><tr><th>Parser</th><th>Records</th><th>Coverage</th><th>Scalar accuracy</th><th>Skills F1</th><th>Experience accuracy</th><th>CGPA MAE</th><th>Overall F1</th></tr></thead><tbody>{parserRows.map((row) => <tr key={row.parser}><td><strong>{row.parser}</strong></td><td>{row.records_predicted}/{row.records_expected}</td><td>{row.record_coverage_percent}%</td><td>{row.scalar_accuracy_percent}%</td><td>{((row.skills?.f1 || 0) * 100).toFixed(1)}%</td><td>{row.experience_accuracy_percent}%</td><td>{row.cgpa_mae ?? '—'}</td><td><span className="score-pill">{row.overall_f1_percent}%</span></td></tr>)}</tbody></table></div>
          </section>

          {summary.retrieval?.queries && (
            <section className="panel">
              <SectionTitle step="D" icon={Search} title="Recruiter benchmark results" text="Per-query retrieval performance using Precision@10, Recall@10, MRR and nDCG@10." />
              <div className="table-scroll"><table><thead><tr><th>Query</th><th>Status</th><th>P@10</th><th>R@10</th><th>MRR</th><th>nDCG@10</th><th>Latency</th></tr></thead><tbody>{summary.retrieval.queries.map((row, index) => <tr key={`${index}-${row.query}`}><td>{row.query}</td><td>{row.status === 'success' ? <Pill tone="success">Success</Pill> : <Pill tone="warning">Failed</Pill>}</td><td>{(row.precision_at_k * 100).toFixed(1)}%</td><td>{(row.recall_at_k * 100).toFixed(1)}%</td><td>{row.mrr.toFixed(3)}</td><td>{(row.ndcg_at_k * 100).toFixed(1)}%</td><td>{row.latency_seconds.toFixed(2)}s</td></tr>)}</tbody></table></div>
            </section>
          )}
        </>
      )}
    </div>
  )
}

function System({ status }) {
  const conversion = status?.conversion || {}
  const checks = [
    ['API service', true, 'FastAPI backend responding'],
    ['PDF → Markdown', Boolean(status?.conversion?.total === 0 || status?.conversion?.converted > 0), conversion?.total ? `${conversion.converted}/${conversion.total} converted` : 'Waiting for uploads'],
    ['Traditional NLP', Boolean(status?.nlp_records), status?.nlp_records ? `${status.nlp_records} records available` : 'Output not generated'],
    ['LLM parser', Boolean(status?.llm_records), status?.llm_records ? `${status.llm_records} records available` : 'Output not generated'],
    ['Evaluation benchmark', Boolean(status?.evaluation?.benchmark_available), status?.evaluation?.benchmark_available ? '90-record benchmark ready' : 'Benchmark files not found'],
  ]

  return (
    <div className="page-stack">
      <section className="hero-panel system-hero"><div><Pill tone="lime"><ServerCog size={13} /> SYSTEM HEALTH</Pill><h1>Your pipeline,<br /><span>visible and explainable.</span></h1><p>Monitor the major HireNex components from the same frontend used by recruiters and evaluators.</p></div><div className="system-icon"><Cpu size={48} /></div></section>
      <section className="panel"><SectionTitle step="01" icon={Activity} title="Runtime checks" text="Current state of the local HireNex stack." /><div className="system-list">{checks.map(([label, ok, detail]) => <div className="system-row" key={label}><div className={`status-dot ${ok ? 'ok' : 'warn'}`} /><div className="system-main"><strong>{label}</strong><span>{detail}</span></div><Pill tone={ok ? 'success' : 'warning'}>{ok ? 'Online' : 'Needs setup'}</Pill></div>)}</div></section>
      <section className="two-column"><div className="panel"><SectionTitle step="02" icon={Layers3} title="Architecture" text="The new interface replaces Streamlit without rewriting your core Python intelligence." /><div className="architecture"><div>React + Vite</div><ArrowRight /><div>FastAPI</div><ArrowRight /><div>Python AI</div><div className="architecture-sub">NLP • Gemini • ChromaDB • Excel</div></div></div><div className="panel"><SectionTitle step="03" icon={Database} title="Dataset state" text="Live data counts from the backend." /><div className="system-metrics"><MetricCard icon={FileText} label="PDF" value={conversion.total || 0} caption="Uploaded resumes" tone="cyan" /><MetricCard icon={FileCheck2} label="Markdown" value={conversion.converted || 0} caption="Converted" tone="violet" /><MetricCard icon={BrainCircuit} label="NLP" value={status?.nlp_records || 0} caption="Records" tone="lime" /><MetricCard icon={Sparkles} label="LLM" value={status?.llm_records || 0} caption="Records" tone="magenta" /></div></div></section>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = React.useState('workspace')
  const [status, setStatus] = React.useState(null)
  const [notice, setNotice] = React.useState(null)
  const [activeJob, setActiveJob] = React.useState(null)
  const [apiOnline, setApiOnline] = React.useState(false)

  const refreshStatus = React.useCallback(async () => {
    try {
      const data = await api('/api/status')
      setStatus(data)
      setApiOnline(true)
    } catch (error) {
      setApiOnline(false)
      setNotice({ type: 'error', text: `Backend unavailable: ${error.message}` })
    }
  }, [])

  React.useEffect(() => {
    refreshStatus()
    const interval = setInterval(refreshStatus, 2200)
    return () => clearInterval(interval)
  }, [refreshStatus])

  const activeJobId = activeJob?.job_id || (typeof activeJob === 'string' ? activeJob : null)
  React.useEffect(() => {
    if (!activeJobId) return undefined
    let cancelled = false
    let finished = false
    let interval = null
    const poll = async () => {
      if (cancelled || finished) return
      try {
        const job = await api(`/api/jobs/${activeJobId}`)
        if (cancelled) return
        setActiveJob(job)
        if (['completed', 'failed'].includes(job.status)) {
          finished = true
          if (interval) clearInterval(interval)
          refreshStatus()
        }
      } catch (error) {
        if (!cancelled) {
          finished = true
          if (interval) clearInterval(interval)
        }
      }
    }
    poll()
    interval = setInterval(poll, 850)
    return () => { cancelled = true; if (interval) clearInterval(interval) }
  }, [activeJobId, refreshStatus])

  const pageTitle = tab === 'workspace' ? 'Workspace' : tab === 'evaluation' ? 'Evaluation' : 'System'

  return (
    <div
      className="app-shell"
      onMouseMove={(event) => {
        const rect = event.currentTarget.getBoundingClientRect()
        event.currentTarget.style.setProperty('--pointer-x', `${event.clientX - rect.left}px`)
        event.currentTarget.style.setProperty('--pointer-y', `${event.clientY - rect.top}px`)
      }}
    >
      <header className="topbar">
        <div className="brand-wrap"><div className="brand-mark"><Sparkles size={20} /></div><div><div className="brand-name">HireNex</div><div className="brand-tag">Intelligent Candidate Screening & Retrieval</div></div></div>
        <nav className="nav-tabs">{[['workspace', 'Workspace', Activity], ['evaluation', 'Evaluation', BarChart3], ['system', 'System', Cpu]].map(([key, label, Icon]) => <button key={key} className={tab === key ? 'nav-tab active' : 'nav-tab'} onClick={() => setTab(key)}><Icon size={15} />{label}</button>)}</nav>
        <div className="api-pill"><span className={`api-dot ${apiOnline ? 'online' : ''}`} />{apiOnline ? 'API online' : 'API offline'}</div>
      </header>

      <main className="content">
        <div className="page-breadcrumb"><span>HireNex</span><ChevronRight size={13} /><strong>{pageTitle}</strong></div>
        {notice && <div className={`global-notice ${notice.type}`}><span>{notice.type === 'error' ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}{notice.text}</span><button onClick={() => setNotice(null)}>×</button></div>}
        {tab === 'workspace' && <Workspace status={status} refreshStatus={refreshStatus} showNotice={notice} setShowNotice={setNotice} activeJob={activeJob} setActiveJob={setActiveJob} />}
        {tab === 'evaluation' && <Evaluation status={status} activeJob={activeJob} setActiveJob={setActiveJob} showNotice={notice} setShowNotice={setNotice} refreshStatus={refreshStatus} />}
        {tab === 'system' && <System status={status} />}
      </main>

      <footer><div><strong>HireNex</strong> • Built for explainable AI-assisted recruitment</div><div>Created by <strong>Ram & Co</strong> • Under the guidance of <strong>Usha Rani</strong></div></footer>
    </div>
  )
}
