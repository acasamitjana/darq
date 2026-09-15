import { useState } from 'react'
import Header from './components/Header'
import FileUploadCard from './components/FileUploadCard'
import ProgressPanel from './components/ProgressPanel'
import DataTable from './components/DataTable'
import { getJobResults, submitDarqJob, waitForJob } from './api/darqApi'

const TABS = [
  { id: 'inputs', label: 'Analysis' },
  { id: 'tables', label: 'Results' },
  { id: 'images', label: 'Images' },
  { id: 'pdf', label: 'Report' },
]

const IDLE_PROGRESS = {
  status: 'idle',
  message: '',
  step: 0,
  totalSteps: 0,
}

function EmptyResults({ text }) {
  return (
    <div className="empty-results">
      <div className="empty-results__icon" aria-hidden="true">i</div>
      <div>{text}</div>
    </div>
  )
}

function PageTitle({ eyebrow, title, description }) {
  return (
    <div className="page-title-row">
      <div>
        <div className="page-eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
    </div>
  )
}

export default function DarqPage() {
  const [activeTab, setActiveTab] = useState('inputs')
  const [subjectId, setSubjectId] = useState('tutorial_subject')
  const [datFile, setDatFile] = useState(null)
  const [mriFile, setMriFile] = useState(null)
  const [segFile, setSegFile] = useState(null)
  const [progress, setProgress] = useState(IDLE_PROGRESS)
  const [results, setResults] = useState(null)
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState('')
  const [isRunning, setIsRunning] = useState(false)

  async function runDarq() {
    setError('')

    if (!datFile || !mriFile || !segFile) {
      setError('Please select the DaTSCAN, MRI / T1w and SynthSeg NIfTI files.')
      return
    }

    setResults(null)
    setJobId(null)
    setIsRunning(true)
    setProgress({
      status: 'uploading',
      message: 'Sending input files to FastAPI...',
      step: 0,
      totalSteps: 0,
    })

    try {
      const created = await submitDarqJob({
        datFile,
        mriFile,
        segFile,
        subjectId,
      })

      setJobId(created.job_id)
      setProgress({
        status: 'queued',
        message: 'Files uploaded. Waiting for the DARQ worker...',
        step: 0,
        totalSteps: 0,
      })

      const finalStatus = await waitForJob(created.job_id, (jobStatus) => {
        const status = jobStatus.status || 'queued'
        const pipelineProgress = jobStatus.progress || {}

        if (status === 'queued') {
          setProgress({
            status: 'queued',
            message: 'Files uploaded. Waiting for the DARQ worker...',
            step: 0,
            totalSteps: pipelineProgress.total_steps || 0,
          })
          return
        }

        if (status === 'running') {
          setProgress({
            status: 'running',
            message: pipelineProgress.message || 'Starting DARQ...',
            step: pipelineProgress.step || 1,
            totalSteps: pipelineProgress.total_steps || 5,
          })
        }
      })

      setProgress({
        status: 'finalizing',
        message: 'Preparing tables, images and PDF report...',
        step: finalStatus.progress?.step || 0,
        totalSteps: finalStatus.progress?.total_steps || 5,
      })

      const jobResults = await getJobResults(created.job_id)
      setResults(jobResults)

      setProgress({
        status: 'completed',
        message: 'DARQ processing completed successfully.',
        step: finalStatus.progress?.total_steps || 5,
        totalSteps: finalStatus.progress?.total_steps || 5,
      })
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : String(runError)
      setError(message)
      setProgress({
        status: 'failed',
        message,
        step: 0,
        totalSteps: 0,
      })
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="app-shell">
      <Header showBackLink />

      <nav className="section-nav" aria-label="DARQ sections">
        <div className="section-nav__inner">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={`section-nav__item ${activeTab === tab.id ? 'section-nav__item--active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      <main className="page-container">
        {activeTab === 'inputs' && (
          <>
            <PageTitle
              eyebrow="DARQ workspace"
              title="New analysis"
              description="Prepare one DaTSCAN study for quantitative processing."
            />

            <section className="workspace-card">
              <div className="form-section">
                <div className="form-section__heading">
                  <span className="form-section__index">01</span>
                  <div>
                    <h2>Subject information</h2>
                    <p>Identify the subject or case that will appear in the generated report.</p>
                  </div>
                </div>

                <div className="subject-field">
                  <label htmlFor="subject-id">Subject / Case ID</label>
                  <input
                    id="subject-id"
                    type="text"
                    value={subjectId}
                    onChange={(event) => setSubjectId(event.target.value)}
                    placeholder="Example: sub-001"
                    disabled={isRunning}
                  />
                </div>
              </div>

              <div className="form-divider" />

              <div className="form-section">
                <div className="form-section__heading">
                  <span className="form-section__index">02</span>
                  <div>
                    <h2>Input imaging</h2>
                    <p>Select the three NIfTI files required by the current DARQ workflow.</p>
                  </div>
                </div>

                <div className="upload-grid">
                  <FileUploadCard
                    id="dat-upload"
                    title="DaTSCAN"
                    description="Nuclear medicine image"
                    file={datFile}
                    onChange={setDatFile}
                    disabled={isRunning}
                  />
                  <FileUploadCard
                    id="mri-upload"
                    title="MRI / T1w"
                    description="Structural MRI image"
                    file={mriFile}
                    onChange={setMriFile}
                    disabled={isRunning}
                  />
                  <FileUploadCard
                    id="seg-upload"
                    title="SynthSeg"
                    description="Segmentation image"
                    file={segFile}
                    onChange={setSegFile}
                    disabled={isRunning}
                  />
                </div>

                <div className="run-area">
                  <div className="run-note">Accepted format: .nii or .nii.gz</div>
                  <button
                    type="button"
                    className="run-button"
                    onClick={runDarq}
                    disabled={isRunning}
                  >
                    {isRunning ? 'DARQ is running…' : 'Run DARQ'}
                  </button>
                </div>
              </div>

              {error && <div className="error-banner">{error}</div>}

              <ProgressPanel
                state={progress}
                subjectId={subjectId}
                />
            </section>
          </>
        )}

        {activeTab === 'tables' && (
          <>
            <PageTitle
              eyebrow="Quantitative output"
              title="Results"
              description="Complete tables generated by the DARQ backend, without recalculating or modifying their values."
            />

            <section className="workspace-card workspace-card--results">
              {results ? (
                <div className="results-page">
                  <DataTable
                    title="Complete SBR table generated by the pipeline"
                    table={results.sbr}
                  />
                  <DataTable
                    title="Complete symmetry table generated by the pipeline"
                    table={results.symmetry}
                  />
                </div>
              ) : (
                <EmptyResults text="Run DARQ to display the complete backend result tables." />
              )}
            </section>
          </>
        )}

        {activeTab === 'images' && (
          <>
            <PageTitle
              eyebrow="Qualitative review"
              title="Images"
              description="MRI, registered DaTSCAN and overlay views generated by the existing DARQ visualization code."
            />

            <section className="workspace-card workspace-card--results">
              {results?.overlay_url ? (
                <div className="image-panel">
                  <img src={results.overlay_url} alt="MRI, registered DaTSCAN and overlay views" />
                </div>
              ) : (
                <EmptyResults text="Run DARQ to display the generated qualitative image panel." />
              )}
            </section>
          </>
        )}

        {activeTab === 'pdf' && (
          <>
            <PageTitle
              eyebrow="Study document"
              title="Report"
              description="Download the DARQ PDF report generated by the existing Python report code."
            />

            <section className="workspace-card workspace-card--results">
              {results?.pdf_url ? (
                <div className="report-download">
                  <div className="report-download__icon" aria-hidden="true">PDF</div>
                  <div className="report-download__content">
                    <div className="section-heading">DARQ PDF report</div>
                    <p>The report content and layout are unchanged in this version.</p>
                  </div>
                  <a className="download-button" href={results.pdf_url} target="_blank" rel="noreferrer">
                    Download report
                  </a>
                </div>
              ) : (
                <EmptyResults text="Run DARQ to generate the PDF report." />
              )}
            </section>
          </>
        )}
      </main>

      <footer className="app-footer">
        <div className="app-footer__inner">
          <span>DARQ · DaTSCAN Quantification</span>
          <span>Research use only</span>
        </div>
      </footer>
    </div>
  )
}
