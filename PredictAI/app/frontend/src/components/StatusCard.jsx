function StatusCard({ label, value, description, icon, danger }) {
  return (
    <div className={`status-card ${danger ? "danger" : ""}`}>
      <div className="card-top">
        <span className="card-icon">{icon}</span>
        <span className="card-label">{label}</span>
      </div>

      <div className="card-value">
        {value}
      </div>

      {description && (
        <div className="card-description">
          {description}
        </div>
      )}
    </div>
  );
}

export default StatusCard;