import DarqPage from './DarqPage'
import HomePage from './HomePage'

function normalizedPath() {
  const path = window.location.pathname.replace(/\/+$/, '')
  return path || '/'
}

export default function App() {
  const path = normalizedPath()

  if (path === '/darq') {
    return <DarqPage />
  }

  return <HomePage />
}
