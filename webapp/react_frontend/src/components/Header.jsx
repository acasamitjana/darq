export default function Header({ showBackLink = false }) {
  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="brand-block">
          {showBackLink && (
            <a className="back-to-platform" href="/" aria-label="Back to platform home">
              <span aria-hidden="true">←</span>
              <span>Pipelines</span>
            </a>
          )}

          {showBackLink && <span className="brand-divider" aria-hidden="true" />}

          <div className="brand-title">DARQ</div>
          <div className="brand-copy">
            <div className="brand-subtitle">DaTSCAN Quantification</div>
            <div className="brand-context">Quantitative neuroimaging workspace</div>
          </div>
        </div>

        <div className="institution-block" aria-label="Institution">
          <span className="institution-name">VICOROB</span>
          <span className="institution-divider" aria-hidden="true" />
          <span>Universitat de Girona</span>
        </div>
      </div>
    </header>
  )
}
