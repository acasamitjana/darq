const PIPELINES = [
  {
    id: 'darq',
    name: 'DARQ',
    subtitle: 'DaTSCAN Quantification',
    description: 'Quantitative processing of DaTSCAN studies using structural MRI and anatomical segmentation.',
    href: '/darq',
  },
]

const RESOURCES = [
  {
    id: 'documentation',
    name: 'Documentation',
    subtitle: 'Technical documentation and tutorials',
    description: 'Open the project documentation generated with Sphinx, including deployment instructions, Docker, SSH access and tutorials.',
    href: '/documentation/',
  },
]

function ArrowIcon() {
  return <span className="portal-card__arrow" aria-hidden="true">→</span>
}

function PipelineIcon() {
  return (
    <div className="pipeline-icon" aria-hidden="true">
      <svg viewBox="0 0 36 36" role="presentation">
        <circle cx="18" cy="18" r="10.5" />
        <path d="M10.8 18h14.4" />
        <path d="M18 10.8v14.4" />
        <path d="M13.4 13.4c2.8-2.8 6.4-2.8 9.2 0" />
        <path d="M13.4 22.6c2.8 2.8 6.4 2.8 9.2 0" />
      </svg>
    </div>
  )
}

function ResourceIcon() {
  return (
    <div className="resource-icon" aria-hidden="true">
      <span />
      <span />
      <span />
    </div>
  )
}

function PipelineCard({ pipeline }) {
  return (
    <a className="portal-card portal-card--pipeline" href={pipeline.href}>
      <div className="portal-card__topline">
        <span className="portal-card__type">Processing pipeline</span>
        <span className="availability-badge">Available</span>
      </div>

      <div className="portal-card__body">
        <div className="portal-card__identity">
          <PipelineIcon />
          <div>
            <h3>{pipeline.name}</h3>
            <div className="portal-card__subtitle">{pipeline.subtitle}</div>
          </div>
        </div>
        <p>{pipeline.description}</p>
      </div>

      <div className="portal-card__action">
        <span>Open pipeline</span>
        <ArrowIcon />
      </div>
    </a>
  )
}

function ResourceCard({ resource }) {
  return (
    <a className="portal-card portal-card--resource" href={resource.href}>
      <div className="portal-card__topline">
        <span className="portal-card__type">Project resource</span>
      </div>

      <div className="portal-card__body">
        <div className="portal-card__identity">
          <ResourceIcon />
          <div>
            <h3>{resource.name}</h3>
            <div className="portal-card__subtitle">{resource.subtitle}</div>
          </div>
        </div>
        <p>{resource.description}</p>
      </div>

      <div className="portal-card__action">
        <span>Open documentation</span>
        <ArrowIcon />
      </div>
    </a>
  )
}

export default function HomePage() {
  return (
    <div className="platform-shell">
      <header className="platform-header">
        <div className="platform-header__inner">
          <div className="platform-brand">
            <div className="platform-brand__title">VICOROB</div>
            <div className="platform-brand__divider" aria-hidden="true" />
            <div>
              <div className="platform-brand__subtitle">Medical Imaging Processing Platform</div>
              <div className="platform-brand__context">Research software workspace</div>
            </div>
          </div>

          <div className="institution-block" aria-label="Institution">
            <span>Universitat de Girona</span>
          </div>
        </div>
      </header>

      <main className="platform-main">
        <section className="platform-hero">
          <div className="page-eyebrow">VICOROB imaging platform</div>
          <h1>Select a workspace</h1>
          <p>
            Access the available medical imaging processing pipelines and the technical resources associated with the platform.
          </p>
        </section>

        <section className="portal-section" aria-labelledby="pipelines-title">
          <div className="portal-section__heading">
            <div>
              <div className="portal-section__index">01</div>
              <div>
                <h2 id="pipelines-title">Processing pipelines</h2>
                <p>Select the workflow required for the study you want to process.</p>
              </div>
            </div>
          </div>

          <div className="portal-grid">
            {PIPELINES.map((pipeline) => (
              <PipelineCard key={pipeline.id} pipeline={pipeline} />
            ))}
          </div>
        </section>

        <section className="portal-section" aria-labelledby="resources-title">
          <div className="portal-section__heading">
            <div>
              <div className="portal-section__index">02</div>
              <div>
                <h2 id="resources-title">Resources</h2>
                <p>Open the technical documentation for deployment, usage and development.</p>
              </div>
            </div>
          </div>

          <div className="portal-grid">
            {RESOURCES.map((resource) => (
              <ResourceCard key={resource.id} resource={resource} />
            ))}
          </div>
        </section>
      </main>

      <footer className="app-footer platform-footer">
        <div className="app-footer__inner">
          <span>VICOROB · Medical Imaging Processing Platform</span>
          <span>Research use only</span>
        </div>
      </footer>
    </div>
  )
}
