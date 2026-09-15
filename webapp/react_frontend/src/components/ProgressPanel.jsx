function statusTitle(status) {
  if (status === 'uploading') return 'Uploading files'
  if (status === 'queued') return 'Waiting for worker'
  if (status === 'running') return 'Processing study'
  if (status === 'finalizing') return 'Preparing results'
  if (status === 'completed') return 'Analysis completed'
  if (status === 'failed') return 'Analysis failed'
  return 'Waiting'
}

function statusLabel(status) {
  if (status === 'uploading') return 'UPLOADING'
  if (status === 'queued') return 'QUEUED'
  if (status === 'running') return 'PROCESSING'
  if (status === 'finalizing') return 'FINALIZING'
  if (status === 'completed') return 'COMPLETED'
  if (status === 'failed') return 'FAILED'
  return 'WAITING'
}

function StepTracker({ step, totalSteps, status }) {
  if (!totalSteps || totalSteps < 1) return null

  return (
    <div className="step-tracker" aria-label="Pipeline progress steps">
      {Array.from({ length: totalSteps }, (_, index) => {
        const stepNumber = index + 1
        const isComplete = status === 'completed' || stepNumber < step
        const isCurrent = status !== 'completed' && stepNumber === step

        return (
          <div
            className={`step-tracker__item ${isComplete ? 'step-tracker__item--complete' : ''} ${isCurrent ? 'step-tracker__item--current' : ''}`}
            key={stepNumber}
          >
            <div className="step-tracker__marker">
              {isComplete ? '✓' : stepNumber}
            </div>
            {index < totalSteps - 1 && <div className="step-tracker__line" />}
          </div>
        )
      })}
    </div>
  )
}

export default function ProgressPanel({ state, subjectId, jobId }) {
  if (!state || state.status === 'idle') return null

  const { status, message, step = 0, totalSteps = 0 } = state
  const completedSteps = status === 'completed' ? totalSteps : Math.max(step - 1, 0)
  const percentage = status === 'completed'
    ? 100
    : totalSteps > 0
      ? Math.max(0, Math.min(100, (completedSteps / totalSteps) * 100))
      : 0

  return (
    <section className={`progress-panel progress-panel--${status}`} aria-live="polite">
      <div className="study-status-strip">
        <div className="study-status-strip__item">
          <span>Subject</span>
          <strong>{subjectId || 'subject'}</strong>
        </div>
        <div className="study-status-strip__item study-status-strip__item--job">
          <span>Job</span>
          <strong title={jobId || ''}>{jobId || 'Pending creation'}</strong>
        </div>
        <div className="study-status-strip__item study-status-strip__item--status">
          <span>Status</span>
          <strong className={`clinical-status clinical-status--${status}`}>
            <i aria-hidden="true" />
            {statusLabel(status)}
          </strong>
        </div>
      </div>

      <div className="progress-panel__body">
        <div className="progress-panel__topline">
          <div>
            <div className="progress-panel__title">{statusTitle(status)}</div>
            <div className="progress-panel__message">{message}</div>
          </div>

          {(status === 'running' || status === 'completed') && (
            <div className="progress-percentage">{Math.round(percentage)}%</div>
          )}
        </div>

        {(status === 'running' || status === 'completed') && (
          <>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${percentage}%` }} />
            </div>

            <div className="progress-meta-row">
              <span>
                {status === 'completed'
                  ? 'All steps completed'
                  : `Step ${step} of ${totalSteps}`}
              </span>
              <span>{completedSteps} of {totalSteps} steps completed</span>
            </div>

            <StepTracker step={step} totalSteps={totalSteps} status={status} />
          </>
        )}
      </div>
    </section>
  )
}
