import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function StatCard({ label, value }) {
  return (
    <div className="rounded-xl bg-white p-4 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-slate-800">{value}</p>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [status, setStatus] = useState({
    progress_percent: 0,
    counts: {},
    total_unique_count: 0,
    done: false,
    error: null,
    is_processing: false
  });
  const [uploading, setUploading] = useState(false);

  const wsUrl = useMemo(() => {
    const cleaned = API_BASE.replace("http://", "").replace("https://", "");
    const protocol = API_BASE.startsWith("https") ? "wss" : "ws";
    return `${protocol}://${cleaned}/status`;
  }, []);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_BASE}/models`);
        if (!response.ok) {
          throw new Error("Failed to load model list.");
        }
        const payload = await response.json();
        const modelNames = payload.models || [];
        setModels(modelNames);
        setSelectedModel(payload.default || modelNames[0] || "");
      } catch (err) {
        setStatus((prev) => ({ ...prev, error: err.message }));
      }
    };
    fetchModels();
  }, []);

  useEffect(() => {
    const ws = new WebSocket(wsUrl);
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setStatus((prev) => ({ ...prev, ...payload }));
      } catch {
        // Ignore malformed payloads.
      }
    };
    return () => ws.close();
  }, [wsUrl]);

  const onUpload = async () => {
    if (!file || !selectedModel) return;
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("model_name", selectedModel);
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData
      });
      if (!response.ok) {
        const body = await response.json();
        throw new Error(body.detail || "Upload failed");
      }
    } catch (err) {
      setStatus((prev) => ({ ...prev, error: err.message }));
    } finally {
      setUploading(false);
    }
  };

  const classEntries = Object.entries(status.counts || {});
  const canDownload = status.done && !status.error;

  return (
    <div className="min-h-screen bg-slate-100 p-6">
      <div className="mx-auto max-w-6xl space-y-6">
        <header className="rounded-2xl bg-slate-900 p-6 text-white shadow">
          <h1 className="text-2xl font-bold">Smart Drone Traffic Analyzer</h1>
          <p className="mt-2 text-sm text-slate-300">Upload drone footage and get unique, tracked vehicle counts.</p>
        </header>

        <section className="grid gap-4 md:grid-cols-3">
          <StatCard label="Unique Vehicles" value={status.total_unique_count || 0} />
          <StatCard label="Progress" value={`${Math.round(status.progress_percent || 0)}%`} />
          <StatCard label="Active Classes" value={classEntries.length} />
        </section>

        <section className="rounded-2xl bg-white p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-800">Upload Video</h2>
          <div className="mt-4 flex flex-col gap-4 md:flex-row md:items-center">
            <input
              type="file"
              accept=".mp4"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
            />
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="block w-full rounded-lg border border-slate-300 px-3 py-2 text-sm md:max-w-xs"
              disabled={uploading || status.is_processing}
            >
              {models.length === 0 && <option value="">No models found</option>}
              {models.map((modelName) => (
                <option key={modelName} value={modelName}>
                  {modelName}
                </option>
              ))}
            </select>
            <button
              onClick={onUpload}
              disabled={!file || !selectedModel || uploading || status.is_processing}
              className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {uploading ? "Uploading..." : "Start Processing"}
            </button>
            {canDownload && (
              <a
                href={`${API_BASE}/download-report`}
                className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-medium text-white"
              >
                Download Report
              </a>
            )}
          </div>

          <div className="mt-4 h-3 w-full rounded-full bg-slate-200">
            <div
              className="h-3 rounded-full bg-blue-600 transition-all"
              style={{ width: `${Math.min(100, Math.max(0, status.progress_percent || 0))}%` }}
            />
          </div>

          {status.error && <p className="mt-4 text-sm text-red-600">{status.error}</p>}
          {selectedModel && (
            <p className="mt-2 text-sm text-slate-600">
              Selected model: <span className="font-medium">{selectedModel}</span>
            </p>
          )}
        </section>

        <section className="grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-800">Vehicle Breakdown</h2>
            <div className="mt-4 space-y-2">
              {classEntries.length === 0 && <p className="text-sm text-slate-500">No vehicles counted yet.</p>}
              {classEntries.map(([name, count]) => (
                <div key={name} className="flex items-center justify-between rounded-lg bg-slate-50 px-4 py-2">
                  <span className="font-medium text-slate-700">{name}</span>
                  <span className="font-semibold text-slate-900">{count}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl bg-white p-6 shadow-sm">
            <h2 className="text-lg font-semibold text-slate-800">Processed Video</h2>
            {canDownload ? (
              <video controls className="mt-4 w-full rounded-lg" src={`${API_BASE}/processed-video`} />
            ) : (
              <div className="mt-4 rounded-lg border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
                Processed video will appear when job completes.
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
