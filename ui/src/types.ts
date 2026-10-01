export type Verification = "verified" | "unverified" | "not_found";

export type LineItem = {
  name: string;
  qty: number | null;
  unit_price: number | null;
  amount: number | null;
};

export type Receipt = {
  store_name: string | null;
  date: string | null;
  line_items: LineItem[];
  subtotal: number | null;
  tax: number | null;
  total: number | null;
  _verification: Record<string, Verification>;
};

export type ApiError = {
  error: { code: string; message: string; request_id: string };
};

export type DeskStatus = "needs_review" | "ready" | "approved";

export type RequestMeta = {
  requestId: string;
  cache: string;
  queueWaitMs: string;
  inferenceMs: string;
  httpStatus: number;
};

export type ClerkOverride = {
  path: string;
  value: string;
};

export type DeskItem = {
  id: string;
  fileName: string;
  file: File;
  previewUrl: string | null;
  receipt: Receipt;
  status: DeskStatus;
  meta: RequestMeta;
  acceptUnverified: boolean;
  overrides: ClerkOverride[];
};
