function ChannelSelector({
  channel,
  setChannel,
  onRun,
  loading,
}) {
  return (
    <div className="control-panel">
      <div>
        <span className="control-label">TELEMETRY CHANNEL</span>

        <select
          value={channel}
          onChange={(e) => setChannel(e.target.value)}
        >
          <option value="A-1">A-1</option>
          <option value="A-2">A-2</option>
          <option value="A-3">A-3</option>
          <option value="A-4">A-4</option>
          <option value="A-5">A-5</option>
        </select>
      </div>

      <button
        className="run-button"
        onClick={onRun}
        disabled={loading}
      >
        {loading ? "ANALYZING..." : "RUN ANALYSIS"}
      </button>
    </div>
  );
}

export default ChannelSelector;