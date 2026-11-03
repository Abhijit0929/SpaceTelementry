import { useEffect, useState } from "react";
import { getChannels } from "../services/api";

function ChannelSelector({
  channel,
  setChannel,
  onRun,
  loading,
}) {
  const [channels, setChannels] = useState([]);
  const [loadingChannels, setLoadingChannels] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getChannels()
      .then((data) => setChannels(data.channels))
      .catch((err) => setError(err.message || "Failed to load channel list."))
      .finally(() => setLoadingChannels(false));
  }, []);

  return (
    <div className="control-panel">
      <div>
        <span className="control-label">
          {loadingChannels ? "LOADING CHANNELS..." : "TELEMETRY CHANNEL"}
        </span>

        <select
          value={channel}
          onChange={(e) => setChannel(e.target.value)}
          disabled={loadingChannels}
        >
          {channels.map((c) => (
            <option key={c.channel} value={c.channel}>
              {c.channel} ({c.spacecraft})
            </option>
          ))}
        </select>
      </div>

      <button
        className="run-button"
        onClick={onRun}
        disabled={loading || loadingChannels}
      >
        {loading ? "ANALYZING..." : "RUN ANALYSIS"}
      </button>

      {error && <span className="inline-error">{error}</span>}
    </div>
  );
}

export default ChannelSelector;