import { useEffect, useMemo, useState } from "react";
import { Check, Copy, FileUp, RotateCcw, Upload } from "lucide-react";
import { ProjectPage, SiteHeader } from "./ProjectStory";
import { errorCopy, extractPdf, ExtractHttpError } from "./api";
import { deriveStatus, statusOrder, sumChecks } from "./status";
import type { DeskItem, DeskStatus, Receipt } from "./types";

function money(value: number | null): string {
  return value === null ? "—" : value.toFixed(2);
}

function chip(status: DeskStatus | string) {
  return <span className={`chip ${status}`}>{status.replace("_", " ")}</span>;
}

export function App() {
  const [page, setPage] = useState(window.location.hash || "#/story");
  useEffect(() => {
    const navigate = () => {
      const next = window.location.hash;
      setPage(["#/story", "#/model", "#/evaluation", "#/feasibility", "#/architecture", "#/extract"].includes(next) ? next : "#/story");
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", navigate);
    navigate();
    return () => window.removeEventListener("hashchange", navigate);
  }, []);

  return (
    <>
      <SiteHeader page={page} />
      <main id="main-content" tabIndex={-1}>
        {page !== "#/extract" && <ProjectPage page={page} />}
        <div hidden={page !== "#/extract"}><ReceiptDesk /></div>
      </main>
    </>
  );
}

function ReceiptDesk() {
  const [items, setItems] = useState<DeskItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [phase, setPhase] = useState<"idle" | "queued" | "running">("idle");
  const [error, setError] = useState<string | null>(null);
  const [lastFile, setLastFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);

  const selected = items.find((item) => item.id === selectedId) ?? null;
  const sorted = useMemo(
    () => [...items].sort((a, b) => statusOrder(a.status) - statusOrder(b.status)),
    [items],
  );
  const approvedTotal = items
    .filter((item) => item.status === "approved")
    .reduce((sum, item) => sum + (item.receipt.total ?? 0), 0);
  const reviewCount = items.filter((item) => item.status === "needs_review").length;

  async function upload(file: File) {
    if (busy) return;
    setLastFile(file);
    setError(null);
    setElapsed(0);
    setBusy(true);
    setPhase("queued");
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed(Date.now() - started), 200);
    try {
      setPhase("running");
      const { receipt, meta } = await extractPdf(file);
      const previewUrl = await import("./preview").then(({ renderFirstPage }) => renderFirstPage(file)).catch(() => "");
      const item: DeskItem = {
        id: crypto.randomUUID(),
        fileName: file.name,
        file,
        previewUrl,
        receipt,
        status: deriveStatus(receipt),
        meta,
        acceptUnverified: false,
        overrides: [],
      };
      setItems((current) => [item, ...current]);
      setSelectedId(item.id);
    } catch (err) {
      setError(err instanceof ExtractHttpError ? errorCopy(err) : "Extraction failed.");
    } finally {
      window.clearInterval(timer);
      setBusy(false);
      setPhase("idle");
    }
  }

  async function useSample() {
    try {
      const response = await fetch("/samples/receipt_ok.pdf");
      if (!response.ok) throw new Error("Sample unavailable");
      const blob = await response.blob();
      await upload(new File([blob], "receipt_ok.pdf", { type: "application/pdf" }));
    } catch { setError("The sample receipt could not be loaded."); }
  }

  function patch(id: string, update: Partial<DeskItem>) {
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...update } : item)));
  }

  function approve(item: DeskItem) {
    if (item.status !== "ready" && !item.acceptUnverified) return;
    patch(item.id, { status: "approved" });
  }

  function overrideField(item: DeskItem, path: string, value: string) {
    const overrides = [...item.overrides.filter((row) => row.path !== path), { path, value }];
    patch(item.id, { overrides });
  }

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <p className="eyebrow">06 / The receipt desk</p>
          <h1>Receipt desk</h1>
          <p>
            The document, the extracted data, and the evidence behind it.
          </p>
        </div>
      </header>

      <section className="summary" aria-label="Session summary">
        <article>
          <strong>{items.length}</strong>
          <span>receipts in this session</span>
        </article>
        <article>
          <strong>{money(approvedTotal)}</strong>
          <span>sum of approved totals</span>
        </article>
        <article>
          <strong>{reviewCount}</strong>
          <span>still needing review</span>
        </article>
      </section>

      <section
        className={`drop${over ? " over" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(event) => {
          event.preventDefault();
          setOver(false);
          const file = event.dataTransfer.files[0];
          if (file) void upload(file);
        }}
      >
        <Upload className="upload-symbol" size={28} />
        <h2>Drop a receipt PDF</h2>
        <p>PDF only · Up to 10 MB · Up to 5 pages by default</p>
        <div className="actions">
          <label className="btn">
            <FileUp size={17} /> Choose PDF
            <input
              type="file"
              accept="application/pdf"
              className="file-input"
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void upload(file);
                event.target.value = "";
              }}
            />
          </label>
          <button className="btn secondary" type="button" disabled={busy} onClick={() => void useSample()}>
            Use sample receipt
          </button>
          {lastFile && error ? (
            <button className="btn ghost" type="button" disabled={busy} onClick={() => void upload(lastFile)}>
              <RotateCcw size={16} /> Retry
            </button>
          ) : null}
        </div>
        {busy ? (
          <p>
            {phase === "queued" ? "Queued" : "Running"} · {(elapsed / 1000).toFixed(1)}s
            {lastFile ? ` · ${lastFile.name} (${(lastFile.size / 1024).toFixed(1)} KB)` : ""}
          </p>
        ) : lastFile ? (
          <p>
            {lastFile.name} · {(lastFile.size / 1024).toFixed(1)} KB
          </p>
        ) : null}
        {error ? <div className="banner" role="alert">{error}</div> : null}
      </section>

      {items.length === 0 ? (
        <p className="empty">
          No receipts in this session. Checked responses may be cached for 24 hours by default. Browser session state is lost on refresh.
        </p>
      ) : (
        <div className="inbox">
          {sorted.map((item) => (
            <button
              key={item.id}
              className={`card${item.id === selectedId ? " active" : ""}`}
              type="button"
              onClick={() => setSelectedId(item.id)}
            >
              <div>
                <strong>{item.receipt.store_name ?? "Unknown store"}</strong>
                <div>
                  {item.receipt.date ?? "No date"} · {money(item.receipt.total)}
                </div>
              </div>
              {chip(item.status)}
            </button>
          ))}
        </div>
      )}

      {selected ? <Review item={selected} onApprove={approve} onPatch={patch} onOverride={overrideField} /> : null}
    </div>
  );
}

function Review({
  item,
  onApprove,
  onPatch,
  onOverride,
}: {
  item: DeskItem;
  onApprove: (item: DeskItem) => void;
  onPatch: (id: string, update: Partial<DeskItem>) => void;
  onOverride: (item: DeskItem, path: string, value: string) => void;
}) {
  const banners = sumChecks(item.receipt);
  const canApprove = item.status === "ready" || item.acceptUnverified;
  return (
    <section className="review">
      <figure className="preview">
        <figcaption>First page · {item.fileName}</figcaption>
        {item.previewUrl ? <img src={item.previewUrl} alt={`First page of ${item.fileName}`} /> : <p>No preview</p>}
      </figure>
      <div className="fields">
        {banners.map((message) => (
          <div className="banner" key={message} role="status">
            {message}
          </div>
        ))}
        <Field
          label="Store"
          path="store_name"
          value={item.receipt.store_name}
          receipt={item.receipt}
          item={item}
          onOverride={onOverride}
        />
        <Field
          label="Date"
          path="date"
          value={item.receipt.date}
          receipt={item.receipt}
          item={item}
          onOverride={onOverride}
        />
        <table>
          <thead>
            <tr>
              <th>Item</th>
              <th>Qty</th>
              <th>Unit</th>
              <th>Amount</th>
            </tr>
          </thead>
          <tbody>
            {item.receipt.line_items.map((row, index) => (
              <tr key={`${row.name}-${index}`}>
                <td>
                  {row.name} {chip(item.receipt._verification[`line_items.${index}.name`] ?? "not_found")}
                </td>
                <td>{row.qty ?? "—"}</td>
                <td>{money(row.unit_price)}</td>
                <td>{money(row.amount)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <Field label="Subtotal" path="subtotal" value={item.receipt.subtotal} receipt={item.receipt} item={item} onOverride={onOverride} />
        <Field label="Tax" path="tax" value={item.receipt.tax} receipt={item.receipt} item={item} onOverride={onOverride} />
        <Field label="Total" path="total" value={item.receipt.total} receipt={item.receipt} item={item} onOverride={onOverride} />

        {item.status === "needs_review" ? (
          <label>
            <input
              type="checkbox"
              checked={item.acceptUnverified}
              onChange={(event) => onPatch(item.id, { acceptUnverified: event.target.checked })}
            />{" "}
            I accept the unverified fields
          </label>
        ) : null}

        <div className="actions">
          <button className="btn" type="button" disabled={!canApprove || item.status === "approved"} onClick={() => onApprove(item)}>
            <Check size={17} /> {item.status === "approved" ? "Approved" : "Approve"}
          </button>
        </div>

        <div className="inspector">
          <details>
            <summary>Technical details</summary>
            <p>
              request {item.meta.requestId || "—"} · cache {item.meta.cache} · queue {item.meta.queueWaitMs} ms ·
              inference {item.meta.inferenceMs} ms · HTTP {item.meta.httpStatus}
            </p>
            <button
              className="btn ghost"
              type="button"
              onClick={() => void navigator.clipboard.writeText(JSON.stringify(item.receipt, null, 2))}
            >
              <Copy size={16} /> Copy JSON
            </button>
            <pre>{JSON.stringify(item.receipt, null, 2)}</pre>
          </details>
        </div>
      </div>
    </section>
  );
}

function Field({
  label,
  path,
  value,
  receipt,
  item,
  onOverride,
}: {
  label: string;
  path: string;
  value: string | number | null;
  receipt: Receipt;
  item: DeskItem;
  onOverride: (item: DeskItem, path: string, value: string) => void;
}) {
  const status = receipt._verification[path] ?? "not_found";
  const override = item.overrides.find((row) => row.path === path);
  const isMoney = typeof value === "number" || path === "subtotal" || path === "tax" || path === "total";
  const shown =
    override?.value ??
    (status === "unverified" && isMoney && !item.acceptUnverified ? "" : value === null ? "" : String(value));
  return (
    <div className="field-row">
      <label>
        {label}
        <input
          aria-label={label}
          value={shown}
          onChange={(event) => onOverride(item, path, event.target.value)}
        />
      </label>
      {chip(status)}
    </div>
  );
}
