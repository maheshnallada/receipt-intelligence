# Story UI validation

## Deployed frontend

[Open the frontend](https://receipt-intelligence-chi.vercel.app).
This deployment hosts the frontend only; live extraction requires a separately
running API and GPU inference backend. The checks below were performed locally
and do not establish validation of this hosted deployment.

## Scope

Six hash-routed views: story, model, evaluation, feasibility, architecture,
and extraction. Browser checks used the local FastAPI dummy backend with an
in-memory cache, not Qwen inference. No new GPU benchmark or Docker validation
was performed.

## Results

- `cd ui && npx tsc --noEmit && npm run build`: passed.
- Editor diagnostics for the modified React and CSS files: no errors.
- All six routes at 360, 390, 768, 1440, and 1920 px: no horizontal overflow.
- Desktop and mobile screenshots captured for visual inspection.
- F1/precision selection and pre/post-grounding toggling: passed.
- Cost calculator: default estimates $3.40/$5.81; a $1 hourly rate changes them
  to $6.46/$11.04 for zero-shot/fine-tuned respectively.
- Sample PDF extraction, rendered preview, field verification, and approval:
  passed against the dummy API. Approved total was 13.50.
- Navigation away from the receipt desk and back preserves the receipt.
- Mobile menu opens, navigates, and closes.
- Invalid PDF content produces the visible corrupt/unreadable error.
- Skip-to-content focuses the main element without changing the current route.
- Embedded architecture loads and contains the model path without horizontal
  overflow at the tested desktop size.
- Both public evaluation JSON files exactly match their notebook originals.

The initial JavaScript bundle is approximately 192 KB before compression;
the PDF preview module is lazy-loaded. The Google Fonts stylesheet requires
network access; local fallback fonts remain available.

## Runtime evidence update (2026-10-01)

The evaluation and feasibility views now display the saved API response and
selectable completion/timeout runs from
[api_validation.json](../ui/public/evidence/api_validation.json).
The current response is 25.897 seconds; the 31.619-second observation remains
historical notebook evidence. No new GPU execution was performed locally.

- TypeScript check and Vite production build passed; initial JavaScript is
  approximately 195.75 KB before compression, with PDF preview still lazy-loaded.
- Both views passed browser checks at 360, 390, 768, and 1440 px: five request
  rows, working run selection, visible HTTP-status columns, no page overflow.
- Narrow tables scroll within their section instead of hiding HTTP statuses.
- Desktop/mobile runtime-section screenshots captured; desktop headline
  overflow check passed. Runtime JSON download returned HTTP 200 with both runs.
- Every UI request row matches its raw source capture; response timing and
  adapter identity match the current saved artifacts. Report links resolve.
- Completion-run timeout remains unknown in the snapshot; the saved 60-second
  settings are associated only with the timeout run. Live backend status is
  separate from these saved observations.
- CORD and SROIE public snapshots remain byte-identical to their notebook
  sources. No benchmark or cost-model inputs changed.

## Diagram delivery receipt

```text
diagram_type: architecture
output: /Users/uma.maheswara/AgenticAI/FineTuning_VLM/ui/public/architecture.html
specification_sha256: 550fd7c2eff0fe8d46fe972f6e4ca5975938c6fee48cd66213e941c9fc45fd65
artifact_sha256: 94d921f88685b6898fe18946ab80f849b85da991cebdd292e105b2272b77c442
validation: 9/9 showcase, 0 errors, 0 warnings
browser_evidence: passed
visual_review: passed
correction_rounds: 1
```

The accepted artifact passed automated containment at 1440x900, 1600x1000,
1920x1080, and 2048x1320. Light/dark screenshot evidence and the artifact-bound
browser receipt are under [architecture-validation](architecture-validation/).
Deterministic validation, automated browser evidence, and screenshot review
are separate checks. The earlier tall candidate failed desktop containment;
the accepted landscape candidate replaces it.

Archify reported version 3.0.1 available; the installed skill was not updated.