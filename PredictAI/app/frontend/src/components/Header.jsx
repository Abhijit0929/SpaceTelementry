function Header({ healthy }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">P</div>

        <div>
          <h1>PREDICT<span>AI</span></h1>
          <p>SPACECRAFT TELEMETRY INTELLIGENCE</p>
        </div>
      </div>

      <div className="system-status">
        <span className={`status-dot ${healthy ? "online" : "offline"}`} />
        <div>
          <strong>{healthy ? "SYSTEM ONLINE" : "SYSTEM OFFLINE"}</strong>
          <small>Mission Control</small>
        </div>
      </div>
    </header>
  );
}

export default Header;