import type { ApiError, Receipt, RequestMeta } from "./types";

export class ExtractHttpError extends Error {
  status: number;
  body: ApiError | null;

  constructor(status: number, body: ApiError | null, fallback: string) {
    super(body?.error.message ?? fallback);
    this.status = status;
    this.body = body;
  }
}

export function errorCopy(error: ExtractHttpError): string {
  const code = error.body?.error.code;
  switch (error.status) {
    case 413:
      return "This file is larger than the 10 MB limit.";
    case 422:
      return "This PDF has more than 5 pages.";
    case 400:
      return code === "PDF_PASSWORD_PROTECTED"
        ? "This PDF is password-protected. Remove the password and try again."
        : "This PDF is corrupt or unreadable.";
    case 415:
      return "Only PDF files are accepted.";
    case 429:
      return "The GPU queue is full. Wait a moment and retry.";
    case 503:
      return "The queue wait timed out. Retry in a few seconds.";
    case 504:
      return "Inference timed out on a page.";
    default:
      return error.message || "Extraction failed.";
  }
}

export async function extractPdf(file: File, forceRefresh = false): Promise<{
  receipt: Receipt;
  meta: RequestMeta;
}> {
  const body = new FormData();
  body.append("file", file);
  body.append("force_refresh", forceRefresh ? "true" : "false");
  const response = await fetch("/api/extract", { method: "POST", body });
  if (!response.ok) {
    let parsed: ApiError | null = null;
    try {
      parsed = (await response.json()) as ApiError;
    } catch {
      parsed = null;
    }
    throw new ExtractHttpError(response.status, parsed, response.statusText);
  }
  const receipt = (await response.json()) as Receipt;
  return {
    receipt,
    meta: {
      requestId: response.headers.get("X-Request-Id") ?? "",
      cache: response.headers.get("X-Cache") ?? "MISS",
      queueWaitMs: response.headers.get("X-Queue-Wait-Ms") ?? "0",
      inferenceMs: response.headers.get("X-Inference-Ms") ?? "0",
      httpStatus: response.status,
    },
  };
}

export async function waitReady(timeoutMs = 30_000): Promise<boolean> {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const response = await fetch("/api/ready");
      if (response.ok) return true;
    } catch {
      /* retry */
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  return false;
}
