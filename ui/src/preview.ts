import * as pdfjs from "pdfjs-dist";
import workerSrc from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerSrc;

async function render(file: File): Promise<string | null> {
  const data = await file.arrayBuffer();
  const doc = await pdfjs.getDocument({ data }).promise;
  const page = await doc.getPage(1);
  const viewport = page.getViewport({ scale: 1.6 });
  const canvas = document.createElement("canvas");
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  await page.render({ canvasContext: context, viewport }).promise;
  return canvas.toDataURL("image/png");
}

export async function renderFirstPage(file: File): Promise<string | null> {
  try {
    return await Promise.race([
      render(file),
      new Promise<string | null>((resolve) => {
        window.setTimeout(() => resolve(null), 5000);
      }),
    ]);
  } catch {
    return null;
  }
}
