export default function DataTable({ title, table }) {
  if (!table || !table.columns?.length) {
    return (
      <section className="table-card">
        <div className="section-heading">{title}</div>
        <div className="empty-state">No data available.</div>
      </section>
    )
  }

  return (
    <section className="table-card">
      <div className="section-heading">{title}</div>
      <div className="table-scroll">
        <table className="results-table">
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((value, columnIndex) => (
                  <td key={`${rowIndex}-${columnIndex}`}>
                    {value === null || value === undefined ? '—' : String(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
