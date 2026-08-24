const POLL_INTERVAL_MS = 2000
const MAX_WAIT_MS = 60 * 60 * 1000

async function readError(response) {
  try {
    const payload = await response.json()
    return payload.detail || payload.message || JSON.stringify(payload)
  } catch {
    return `${response.status} ${response.statusText}`
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options)
  if (!response.ok) {
    throw new Error(await readError(response))
  }
  return response.json()
}

export async function submitDarqJob({ datFile, mriFile, segFile, subjectId }) {
  const formData = new FormData()
  formData.append('dat_file', datFile)
  formData.append('mri_file', mriFile)
  formData.append('seg_file', segFile)
  formData.append('subject_id', subjectId || 'subject')

  return requestJson('/api/jobs/darq', {
    method: 'POST',
    body: formData,
  })
}

export function getJobStatus(jobId) {
  return requestJson(`/api/jobs/${encodeURIComponent(jobId)}`)
}

export function getJobResults(jobId) {
  return requestJson(`/api/jobs/${encodeURIComponent(jobId)}/results`)
}

export async function waitForJob(jobId, onStatus) {
  const startedAt = Date.now()

  while (Date.now() - startedAt <= MAX_WAIT_MS) {
    const status = await getJobStatus(jobId)
    onStatus(status)

    if (status.status === 'completed') {
      return status
    }

    if (status.status === 'failed') {
      throw new Error(status.error || 'DARQ job failed.')
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS))
  }

  throw new Error('DARQ did not finish within 3600 seconds.')
}
