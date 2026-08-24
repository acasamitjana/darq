function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 16V4m0 0-4 4m4-4 4 4M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
    </svg>
  )
}

export default function FileUploadCard({ id, title, description, file, onChange, disabled = false }) {
  return (
    <label
      className={`upload-card ${file ? 'upload-card--selected' : ''} ${disabled ? 'upload-card--disabled' : ''}`}
      htmlFor={id}
    >
      <input
        id={id}
        className="upload-card__input"
        type="file"
        accept=".nii,.nii.gz"
        disabled={disabled}
        onChange={(event) => onChange(event.target.files?.[0] ?? null)}
      />

      <div className="upload-card__mainline">
        <div className="upload-card__icon">
          <UploadIcon />
        </div>

        <div className="upload-card__copy">
          <div className="upload-card__title">{title}</div>
          <div className="upload-card__description">{description}</div>
        </div>

        <span className={`file-state ${file ? 'file-state--ready' : ''}`}>
          {file ? 'READY' : 'REQUIRED'}
        </span>
      </div>

      <div className="upload-card__filebox">
        <span className="upload-card__filename" title={file?.name || ''}>
          {file ? file.name : 'No file selected'}
        </span>
        <span className="upload-card__action">
          {file ? 'Change' : 'Browse'}
        </span>
      </div>
    </label>
  )
}
