import { useEffect, useState } from "react";
import { ArrowDown, ArrowRight, ArrowUpRight, Check, ChevronRight, Cpu, Download, ExternalLink, FileJson2, Layers3, Menu, RefreshCw, ShieldCheck, X } from "lucide-react";

const chapters = [
  ["story", "The story"], ["model", "The model"], ["evaluation", "Evaluation"],
  ["feasibility", "Feasibility"], ["architecture", "Architecture"],
] as const;

type Score = { precision: number; recall: number; f1: number; exact_match: number; labeled: number };
type Fields = Record<"store_name" | "date" | "subtotal" | "tax" | "total", Score> & { line_items: Record<string, Score> };
type Stage = {
  fields: Fields; exact_match_receipt: number; json_validity: number;
  hallucination_after_grounding: number; latency_ms_per_page: { p50: number; p95: number };
  peak_gpu_memory: string; pre_grounding: { fields: Fields; hallucination_rate: number };
};
type Evidence = { n: number; stages: Stage[] };
type RuntimeCapture = {
  api: { backend: string; elapsed_s: number; total: number; verified_fields: number; adapter_revision: string };
  runs: {
    id: string; label: string; provenance: string; queue_timeout_s: number | null;
    requests: { id: number; status: number; cache: string | null; queue_wait_ms: number | null; inference_ms: number | null; elapsed_ms: number }[];
  }[];
};
type Version = { backend: string; model_id: string; adapter_revision: string; schema_version: string; prompt_version: string };
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

export function SiteHeader({ page }: { page: string }) {
  const [open, setOpen] = useState(false);
  const [version, setVersion] = useState<Version | null>(null);
  const [checking, setChecking] = useState(true);
  async function checkApi() {
    setChecking(true);
    try {
      const response = await fetch("/api/version", { signal: AbortSignal.timeout(5000) });
      if (!response.ok) throw new Error("Unavailable");
      const value: Version = await response.json();
      if (typeof value.backend !== "string") throw new Error("Invalid version");
      setVersion(value);
    } catch { setVersion(null); }
    finally { setChecking(false); }
  }
  useEffect(() => { void checkApi(); }, []);
  useEffect(() => {
    setOpen(false);
    document.title = `${chapters.find(([id]) => page === `#/${id}`)?.[1] ?? "Receipt desk"} | Receipt intelligence`;
  }, [page]);
  return <>
    <a className="skip-link" href="#main-content" onClick={event => { event.preventDefault(); document.getElementById("main-content")?.focus(); }}>Skip to content</a>
    <header className="site-header">
      <a href="#/story" className="wordmark" aria-label="Receipt intelligence home"><Layers3 size={25} /><span>receipt<span className="wordmark-light"> / intelligence</span></span></a>
      <button className="icon-button mobile-menu" aria-label={open ? "Close navigation" : "Open navigation"} aria-expanded={open} onClick={() => setOpen(!open)}>{open ? <X /> : <Menu />}</button>
      <nav className={open ? "site-nav is-open" : "site-nav"} aria-label="Main navigation">
        {chapters.map(([id, label]) => <a key={id} href={`#/${id}`} aria-current={page === `#/${id}` ? "page" : undefined}>{label}</a>)}
        <a className="nav-desk" href="#/extract" aria-current={page === "#/extract" ? "page" : undefined}>Receipt desk <ArrowUpRight size={16} /></a>
      </nav>
    </header>
    <div className="environment-bar">
      <span>Qwen2.5-VL-3B <span className="environment-divider">/</span> A fine-tuning case study</span>
      <div className="connection"><span className={`status-dot ${version ? "online" : "offline"}`} /><span>{checking ? "Checking API" : !version ? "API unavailable" : version.backend === "dummy" ? "Demo backend · synthetic output" : `Live backend · ${version.backend} · adapter ${version.adapter_revision}`}</span><button className="icon-button" title="Refresh API status" aria-label="Refresh API status" disabled={checking} onClick={() => void checkApi()}><RefreshCw size={13} /></button></div>
    </div>
  </>;
}

function ChapterHeading({ number, label, title, children }: { number: string; label: string; title: string; children: React.ReactNode }) {
  return <header className="chapter-heading"><p className="eyebrow"><span>{number}</span> {label}</p><h1>{title}</h1><p className="chapter-intro">{children}</p></header>;
}

function NextChapter({ to, label, title }: { to: string; label: string; title: string }) {
  return <a className="next-chapter" href={`#/${to}`}><span><span className="eyebrow">{label}</span><strong>{title}</strong></span><ArrowRight size={30} /></a>;
}

export function ProjectPage({ page }: { page: string }) {
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    fetch("/evidence/full_comparison_pipeline.json", { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error("Missing artifact"); return response.json(); })
      .then((data: Evidence) => { if (data.stages.length !== 2) throw new Error("Invalid artifact"); setEvidence(data); })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, []);
  const metrics = evidence ? <Evaluation evidence={evidence} /> : <p role="status" className="evidence-status">{error ? "The evaluation artifact could not be loaded. Reload this page to retry." : "Loading evaluation artifact..."}</p>;
  return <>
    {page === "#/story" && <Story />}
    {page === "#/model" && <Model />}
    {page === "#/evaluation" && <div className="chapter"><ChapterHeading number="03" label="The evidence" title="A small gain. A real tradeoff.">The same 32 CORD receipts, the same prompt, and the same grounding pipeline. Only the adapter changes.</ChapterHeading>{metrics}<RuntimeEvidence /><NextChapter to="feasibility" label="04 / The operating cost" title="What would it take to run this?" /></div>}
    {page === "#/feasibility" && <Feasibility evidence={evidence} error={error} />}
    {page === "#/architecture" && <Architecture />}
    <footer className="site-footer"><span>Receipt intelligence <span className="footer-slash">/</span> From pixels to accountable data.</span><a href="/evidence/full_comparison_pipeline.json" download>Evaluation artifact <Download size={14} /></a></footer>
  </>;
}

function Story() {
  return <>
    <section className="story-hero">
      <div className="hero-inner"><p className="eyebrow">01 / A document intelligence experiment</p><h1>Receipt<br />intelligence.</h1><p className="hero-copy">A small vision-language model.<br />A receipt full of details.<br />A pipeline that asks for evidence.</p><div className="hero-actions"><a href="#/extract" className="btn">Open receipt desk <ArrowUpRight size={18} /></a><a href="#/model" className="text-link">Follow the experiment <ArrowRight size={17} /></a></div><span className="hero-caption">Pictured: the project's synthetic receipt sample</span></div>
    </section>
    <section className="story-intro content-width"><div><p className="eyebrow">The question</p><h2>Can a 3B model turn a receipt<br className="desktop-break" /> into data you can trust?</h2></div><p>Reading a number is only the beginning. This project fine-tunes Qwen2.5-VL, checks its answers against the document, and exposes what still needs a human decision.</p><ArrowDown className="intro-arrow" size={24} /></section>
    <div className="story-facts content-width"><div><strong>3B</strong><span>base model parameters</span></div><div><strong>32</strong><span>held-out CORD receipts</span></div><div><strong>+1</strong><span>additional exact receipt after tuning</span></div><div><strong>1</strong><span>in-flight GPU job per process</span></div></div>
    <section className="story-chapters content-width"><p className="eyebrow">The experiment, in four chapters</p>{[
      { id: "model", number: "02", title: "Start small. Adapt deliberately.", text: "A 3B vision-language model, a 4-bit base, and a receipt-focused LoRA adapter.", icon: <Cpu /> },
      { id: "evaluation", number: "03", title: "Measure the gain, not the ambition.", text: "One more exact receipt on the frozen test set. Higher latency. The full comparison, including the limitations.", icon: <FileJson2 /> },
      { id: "feasibility", number: "04", title: "Put a price on the tradeoff.", text: "T4 latency, reported memory, and a transparent cost estimate for a serialized service.", icon: <Layers3 /> },
      { id: "architecture", number: "05", title: "Build checks around the model.", text: "Grounding, arithmetic checks, bounded admission, and a versioned cache before the result reaches a reviewer.", icon: <ShieldCheck /> },
    ].map(chapter => <a className="chapter-row" key={chapter.id} href={`#/${chapter.id}`}><span className="chapter-index">{chapter.number}</span><div><h3>{chapter.title}</h3><p>{chapter.text}</p></div><span className="chapter-icon">{chapter.icon}</span><ArrowUpRight size={22} /></a>)}</section>
    <section className="story-conclusion"><div className="content-width"><p className="eyebrow">The principle</p><h2>A plausible answer is not<br />the same as a supported one.</h2><p>Unmatched values become null. Conflicting totals are flagged. A reviewer sees the original document alongside the structured result.</p><a href="#/extract" className="btn light">Try a receipt <ArrowRight size={18} /></a></div></section>
  </>;
}

function Model() {
  return <div className="chapter"><ChapterHeading number="02" label="The model" title="Small enough to adapt.">Qwen2.5-VL-3B-Instruct supplies the visual understanding. QLoRA specializes a small set of trainable weights for structured receipt extraction.</ChapterHeading>
    <div className="model-banner"><Cpu size={40} /><div><span className="eyebrow">Base + adapter</span><h2>Qwen2.5-VL-3B-Instruct</h2><p>Vision-language model / 3B parameters / 4-bit base</p></div><a className="icon-button" href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct" target="_blank" rel="noreferrer" aria-label="Open Qwen model on Hugging Face" title="Base model on Hugging Face"><ExternalLink size={22} /></a></div>
    <section className="editorial-grid"><div><p className="eyebrow">The adaptation</p><h2>Teach the output.<br />Keep the foundation.</h2><p>Parameter-efficient fine-tuning adds a LoRA adapter to the quantized base. The target is receipt JSON: store, date, line items, subtotal, tax, and total.</p><a className="text-link" href="https://huggingface.co/Mahesh-Nallada/vlm-receipt-extraction-lora" target="_blank" rel="noreferrer">Project adapter on Hugging Face <ArrowUpRight size={16} /></a></div><dl className="spec-list"><div><dt>Training approach</dt><dd>4-bit QLoRA / PEFT</dd></div><div><dt>Receipt dataset</dt><dd>CORD v2</dd></div><div><dt>Frozen comparison</dt><dd>32 test receipts, ids 0 through 31</dd></div><div><dt>Notebook training cap</dt><dd>200 examples</dd></div><div><dt>Notebook LoRA settings</dt><dd>Rank 16 / alpha 32 / dropout 0</dd></div><div><dt>Header evaluation</dt><dd>SROIE, reported separately</dd></div></dl></section>
    <p className="note">Notebook settings describe the experiment configuration, not independently verified metadata of the published adapter. The repository includes multiple training runs; the saved comparison artifact is the source for the results shown here.</p>
    <section className="section-block"><div className="section-heading"><div><p className="eyebrow">Service defaults</p><h2>The boundaries are explicit.</h2></div><span className="small-label">Configuration, not live telemetry</span></div><div className="config-grid">{[
      ["10 MB", "Maximum upload"], ["5 pages", "Maximum document"], ["1008 px", "Rendered long edge"], ["790,272", "Maximum vision pixels"], ["8", "Queue depth"], ["60 s", "Queue wait timeout"], ["120 s", "Inference timeout"], ["24 h", "Cache TTL"],
    ].map(([value, label]) => <div key={label}><strong>{value}</strong><span>{label}</span></div>)}</div><p className="note">Environment variables can override these defaults. Backend defaults to dummy; real inference requires the qwen backend and an adapter path. Schema and prompt versions default to 1.0.0; adapter revision defaults to base.</p></section>
    <section className="principle-strip"><ShieldCheck size={26} /><div><h3>The model is a proposal, not the final authority.</h3><p>Parsing, document grounding, and arithmetic validation remain separate from generation. Unsupported output is not silently promoted to verified data.</p></div></section><NextChapter to="evaluation" label="03 / The evidence" title="What changed after fine-tuning?" /></div>;
}

function Evaluation({ evidence }: { evidence: Evidence }) {
  const [metric, setMetric] = useState<"f1" | "precision" | "recall" | "exact_match">("f1");
  const [grounded, setGrounded] = useState(true);
  const [baseline, tuned] = evidence.stages;
  const baselineFields = grounded ? baseline.fields : baseline.pre_grounding.fields;
  const tunedFields = grounded ? tuned.fields : tuned.pre_grounding.fields;
  const rows: [string, Score, Score][] = [
    ["Subtotal", baselineFields.subtotal, tunedFields.subtotal], ["Tax", baselineFields.tax, tunedFields.tax], ["Total", baselineFields.total, tunedFields.total],
    ...["name", "qty", "unit_price", "amount"].map(key => [`Item ${key.replace("_", " ")}`, baselineFields.line_items[key], tunedFields.line_items[key]] as [string, Score, Score]),
    ["Store", baselineFields.store_name, tunedFields.store_name], ["Date", baselineFields.date, tunedFields.date],
  ];
  return <>
    <div className="evidence-meta"><span><span className="status-dot online" /> Saved experiment</span><span>CORD v2 / {evidence.n} test receipts / T4</span><a href="/evidence/full_comparison_pipeline.json" download>Source JSON <Download size={15} /></a></div>
    <div className="result-headlines"><div><span>Exact receipt match</span><strong>{percent(baseline.exact_match_receipt)} <ArrowRight size={22} /> {percent(tuned.exact_match_receipt)}</strong><small>6 of 32 to 7 of 32 receipts</small></div><div><span>Total field F1</span><strong>{percent(baseline.fields.total.f1)} <ArrowRight size={22} /> {percent(tuned.fields.total.f1)}</strong><small>After document grounding</small></div><div className="tradeoff"><span>Median latency / page</span><strong>{(baseline.latency_ms_per_page.p50 / 1000).toFixed(1)}s <ArrowRight size={22} /> {(tuned.latency_ms_per_page.p50 / 1000).toFixed(1)}s</strong><small>More time for a modest quality gain</small></div></div>
    <section className="section-block"><div className="section-heading"><div><p className="eyebrow">Field-level comparison</p><h2>Where the difference appears.</h2></div><label className="toggle-label"><input type="checkbox" checked={grounded} onChange={event => setGrounded(event.target.checked)} /> After grounding</label></div><div className="metric-toolbar"><div className="segmented" aria-label="Evaluation metric">{([['f1', 'F1'], ['precision', 'Precision'], ['recall', 'Recall'], ['exact_match', 'Exact match']] as const).map(([key, label]) => <button key={key} aria-pressed={metric === key} onClick={() => setMetric(key)}>{label}</button>)}</div><div className="chart-legend"><span><i className="baseline-key" /> Zero-shot</span><span><i className="tuned-key" /> Fine-tuned</span></div></div>
      <div className="score-table-wrap"><table className="score-table"><thead><tr><th scope="col">Field</th><th scope="col">Comparison</th><th scope="col">Zero-shot</th><th scope="col">Fine-tuned</th><th scope="col">Change</th></tr></thead><tbody>{rows.map(([label, before, after]) => <tr key={label}><th scope="row">{label}</th><td><div className="bar-pair" aria-hidden="true"><span style={{ width: `${before.labeled ? before[metric] * 100 : 0}%` }} /><span style={{ width: `${after.labeled ? after[metric] * 100 : 0}%` }} /></div></td><td>{before.labeled ? percent(before[metric]) : "N/A"}</td><td>{after.labeled ? percent(after[metric]) : "N/A"}</td><td className={after[metric] > before[metric] ? "positive" : ""}>{!before.labeled ? "No labels" : Math.abs(after[metric] - before[metric]) < .000001 ? "Unchanged" : `${after[metric] > before[metric] ? "+" : ""}${((after[metric] - before[metric]) * 100).toFixed(2)} pp`}</td></tr>)}</tbody></table></div><p className="note">Store and date have no gold labels in this CORD slice, so they are N/A. Unlabeled values are ignored; this is not evidence of header extraction accuracy. Changes are percentage points.</p></section>
    <section className="editorial-grid grounding-section"><div><p className="eyebrow">What grounding changes</p><h2>Evidence before confidence.</h2><p>Both runs report the same reduction in unsupported values. Grounding uses CORD annotation text in this evaluation, not production OCR.</p></div><div><div className="grounding-number"><span>{percent(tuned.pre_grounding.hallucination_rate)}</span><ArrowRight /><strong>{percent(tuned.hallucination_after_grounding)}</strong></div><p className="small-label">Reported unsupported-value rate, before to after grounding</p><p className="note">Zero on this metric is not a zero-hallucination guarantee. Derived unit prices are excluded from its denominator; text presence does not prove field semantics.</p></div></section>
    <section className="section-block"><p className="eyebrow">Read the result with its limits</p><div className="limits-grid"><div><span>01</span><h3>A small test set</h3><p>One additional exact receipt is a useful observation, not a claim of statistically significant improvement.</p></div><div><span>02</span><h3>JSON validity is reported</h3><p>The artifact reports 100% pipeline JSON validity. The notebook summary fixes this value; it is not an independent raw-generation validity test.</p></div><div><span>03</span><h3>Headers are a separate task</h3><p>SROIE evaluates 361 receipts with 242 labeled dates. Header scores do not improve between its compared runs and are not pooled with CORD.</p><a className="text-link" href="/evidence/sroie_header_metrics.json" download>SROIE artifact <Download size={14} /></a></div></div></section>
  </>;
}

function RuntimeEvidence() {
  const [capture, setCapture] = useState<RuntimeCapture | null>(null);
  const [error, setError] = useState(false);
  const [selected, setSelected] = useState("completion");
  useEffect(() => {
    const controller = new AbortController();
    fetch("/evidence/api_validation.json", { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error("Missing runtime evidence"); return response.json(); })
      .then((data: RuntimeCapture) => {
        if (!data.api || !Array.isArray(data.runs) || data.runs.length !== 2) throw new Error("Invalid runtime evidence");
        setCapture(data);
      })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, []);
  const run = capture?.runs.find(item => item.id === selected);
  const seconds = (value: number | null) => value === null ? "Not reported" : `${(value / 1000).toFixed(2)} s`;
  return <section className="section-block runtime-evidence" aria-label="Kaggle API evidence">
    <div className="section-heading"><div><p className="eyebrow">Separate runtime evidence</p><h2>Real API. Measured queue behavior.</h2></div><a className="text-link" href="/evidence/api_validation.json" download>Runtime capture <Download size={15} /></a></div>
    {!capture || !run ? <p role="status">{error ? "Runtime evidence unavailable. Reload to retry." : "Loading saved API evidence..."}</p> : <>
      <div className="result-headlines"><div><span>Kaggle API / {capture.api.backend}</span><strong>200 / MISS</strong><small>{capture.api.elapsed_s.toFixed(3)} s for the synthetic receipt</small></div><div><span>Checked response</span><strong>{capture.api.verified_fields} verified fields</strong><small>Total {capture.api.total.toFixed(2)}; schema and arithmetic checks pass</small></div><div><span>Concurrent completion run</span><strong>5 of 5 / HTTP 200</strong><small>Last completion 125.99 s; all cache misses</small></div></div>
      <p className="note">Saved Kaggle execution, not the currently connected backend. The API sample has a PDF text layer; it does not validate OCR fallback or general accuracy. These runs are separate from CORD latency and cost estimates.</p>
      <div className="metric-toolbar"><div className="segmented" aria-label="Concurrency capture">{capture.runs.map(item => <button key={item.id} aria-pressed={selected === item.id} onClick={() => setSelected(item.id)}>{item.label}</button>)}</div><span className="small-label">{run.requests.filter(item => item.status === 200).length} successful / {run.requests.filter(item => item.status === 503).length} HTTP 503</span></div>
      <div className="score-table-wrap"><table className="score-table runtime-table"><caption>{run.label}: saved request timings</caption><thead><tr><th scope="col">Request</th><th scope="col">HTTP / cache</th><th scope="col">Queue wait</th><th scope="col">Processing</th><th scope="col">Elapsed</th></tr></thead><tbody>{run.requests.map(item => <tr key={item.id}><th scope="row">{item.id}</th><td>{item.status} / {item.cache ?? "Not reported"}</td><td>{seconds(item.queue_wait_ms)}</td><td>{seconds(item.inference_ms)}</td><td>{seconds(item.elapsed_ms)}</td></tr>)}</tbody></table></div>
      <p className="note">{run.provenance}</p>
      <p>Increasing waits align with sequential processing. In the 60-second run, two requests return 503 near the queue limit while three complete. Queue timeout limits waiting, not total request duration.</p>
      <p className="note">Client overlap and timings support the queue fix, but are not GPU-forward instrumentation. The captures do not establish production capacity, 429 overflow, or health responsiveness under GPU load. Local regression tests cover overflow, queue timeout, health responsiveness, and cancellation.</p>
    </>}
  </section>;
}

function Feasibility({ evidence, error }: { evidence: Evidence | null; error: boolean }) {
  const [hourly, setHourly] = useState("0.526");
  const rate = Number(hourly);
  const validRate = hourly !== "" && Number.isFinite(rate) && rate >= 0 && rate <= 1000;
  return <div className="chapter"><ChapterHeading number="04" label="Feasibility" title="The gain has an operating cost.">Serialized inference is predictable, but not fast. The fine-tuned run takes longer per page, even though the reported memory peak is unchanged.</ChapterHeading>
    {evidence ? <><div className="latency-comparison"><div className="section-heading"><h2>Latency on the recorded T4 run</h2><span className="small-label">Seconds per page / lower is better</span></div>{evidence.stages.map((stage, index) => <div className="latency-row" key={index}><span>{index ? "Fine-tuned" : "Zero-shot"}</span><div className={`latency-bar ${index ? "tuned" : ""}`} style={{ width: `${stage.latency_ms_per_page.p95 / evidence.stages[1].latency_ms_per_page.p95 * 100}%` }}><span>p50 {(stage.latency_ms_per_page.p50 / 1000).toFixed(1)}s</span><strong>p95 {(stage.latency_ms_per_page.p95 / 1000).toFixed(1)}s</strong></div></div>)}</div>
    <section className="editorial-grid"><div><p className="eyebrow">Memory, carefully stated</p><h2>{evidence.stages[1].peak_gpu_memory}</h2><p>Reported peak allocated memory in both runs, measured by PyTorch. This is not total GPU board memory or a production capacity guarantee.</p><p className="note">Allocator, offload, model loading, and concurrent work affect the real footprint. Do not mix this measurement with older runs.</p></div><div><p className="eyebrow">Deployment posture</p><h2>One process. One GPU slot.</h2><p>FastAPI loads the model once. A bounded queue protects the in-process inference slot; repeat requests can bypass inference through the versioned cache.</p><p className="note">Multiple workers or replicas create separate slots and model copies. This is not a distributed GPU lock. Production load and long-document behavior still need validation.</p></div></section>
    <section className="cost-calculator section-block"><div className="section-heading"><div><p className="eyebrow">Illustrative, not a cloud quote</p><h2>The cost of 1,000 pages.</h2></div><label className="rate-input">Assumed GPU price, USD / hour<input type="number" min="0" max="1000" step="0.001" value={hourly} onChange={event => setHourly(event.target.value)} aria-invalid={!validRate} /></label></div>{!validRate && <p role="alert">Enter an hourly rate from 0 to 1,000.</p>}<div className="cost-results">{evidence.stages.map((stage, index) => <div key={index}><span>{index ? "Fine-tuned" : "Zero-shot"}</span><strong>{validRate ? `$${(stage.latency_ms_per_page.p50 / 1000 * 1000 / 3600 * rate).toFixed(2)}` : "N/A"}</strong><small>~{Math.round(3600 / (stage.latency_ms_per_page.p50 / 1000))} pages / hour at p50</small></div>)}</div><p className="formula">p50 seconds per page × 1,000 ÷ 3,600 × hourly rate</p><p className="note">This is a median-based proxy, not measured average throughput or a bill. It excludes idle capacity, storage, networking, retries, and multi-page variance. The initial $0.526/hour is an assumption, not a verified current price.</p></section></> : <p role="status">{error ? "Evaluation artifact unavailable; cost estimates cannot be calculated." : "Loading measured latency..."}</p>}
    <RuntimeEvidence />
    <section className="readiness"><p className="eyebrow">The next production gates</p><h2>Promising prototype. Not a blank cheque.</h2><ul><li><Check size={18} /> Frozen comparison and pinned-adapter Kaggle API smoke test recorded.</li><li><Check size={18} /> Queue timing captures show five completions and timeout-consistent load shedding.</li><li><ChevronRight size={18} /> Retain matching settings and version metadata for the completion run.</li><li><ChevronRight size={18} /> Expand labeled tests, scanned-document OCR checks, and multi-page validation.</li><li><ChevronRight size={18} /> Measure full-board GPU memory and validate Docker plus the real-model UI path.</li></ul></section><NextChapter to="architecture" label="05 / The system" title="How the checks fit together." /></div>;
}

function Architecture() {
  return <div className="chapter architecture-chapter"><ChapterHeading number="05" label="Architecture" title="A pipeline built around evidence.">The request stays synchronous. The expensive path is serialized. Generation proposes the values; the document and domain rules decide what survives.</ChapterHeading><div className="section-heading diagram-heading"><span className="small-label">Implementation view / real-model path</span><a className="text-link" href="/architecture.html" target="_blank" rel="noreferrer">Open interactive diagram <ExternalLink size={16} /></a></div><iframe className="architecture-frame" src="/architecture.html" title="Interactive receipt extraction architecture" loading="lazy" /><div className="architecture-notes"><div><span className="eyebrow">01 / Fast path</span><h3>Cache before GPU admission.</h3><p>A document hash plus adapter, schema, and prompt versions identifies a cached result. A hit returns without entering the inference slot.</p></div><div><span className="eyebrow">02 / Model path</span><h3>One page at a time.</h3><p>Render with PyMuPDF, generate receipt JSON, parse, and allow one model repair retry. Ground each page against PDF text or Tesseract OCR.</p></div><div><span className="eyebrow">03 / Domain boundary</span><h3>Merge, re-ground, check.</h3><p>Merge pages, re-ground against combined text, run arithmetic sanity checks, then cache the checked response with verification flags.</p></div></div><p className="note">Traced from app/application/extract_receipt.py and app/main.py. The diagram emphasizes the cache-miss path; OCR fallback, cache writeback, error responses, and optional Logfire spans are described here rather than drawn as extra routes. In dummy mode the inference adapter is synthetic.</p><NextChapter to="extract" label="06 / The receipt desk" title="Put a document through the pipeline." /></div>;
}