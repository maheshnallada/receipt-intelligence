import type { DeskStatus, Receipt } from "./types";

const TOLERANCE = 0.05;

function sumsMatch(left: number | null, right: number | null): boolean {
  if (left === null || right === null) return true;
  return Math.abs(left - right) <= TOLERANCE;
}

export function sumChecks(receipt: Receipt): string[] {
  const messages: string[] = [];
  const itemSum = receipt.line_items
    .map((item) => item.amount)
    .filter((value): value is number => value !== null)
    .reduce((acc, value) => acc + value, 0);
  const hasAmounts = receipt.line_items.some((item) => item.amount !== null);
  if (hasAmounts && receipt.subtotal !== null && !sumsMatch(itemSum, receipt.subtotal)) {
    messages.push("Line items do not add up to the subtotal.");
  }
  if (receipt.subtotal !== null && receipt.total !== null) {
    const expected = receipt.subtotal + (receipt.tax ?? 0);
    if (!sumsMatch(expected, receipt.total)) {
      messages.push("Subtotal plus tax does not equal the total.");
    }
  }
  return messages;
}

export function deriveStatus(receipt: Receipt): Exclude<DeskStatus, "approved"> {
  const verification = receipt._verification ?? {};
  const flagged = Object.values(verification).some((value) => value !== "verified");
  const requiredOk = (["store_name", "date", "total"] as const).every(
    (field) => verification[field] === "verified",
  );
  if (flagged || sumChecks(receipt).length > 0 || !requiredOk) {
    return "needs_review";
  }
  return "ready";
}

export function statusOrder(status: DeskStatus): number {
  return { needs_review: 0, ready: 1, approved: 2 }[status];
}
