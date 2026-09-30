/** The largest side a photo is sent at (the server scales to 2048 px anyway). */
const MAX_SIDE = 2048;
/** PNG and WebP may be cut-outs with transparency, which TRELLIS.2 uses as the mask; kept smaller. */
const MAX_SIDE_WITH_ALPHA = 1536;

/**
 * Reads a photo for the AI panel: turned upright (phones store their rotation), scaled down, and
 * encoded as a data URL. JPEG stays JPEG; PNG and WebP become PNG so transparency survives.
 */
export async function photoToDataUrl(file: File): Promise<string> {
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    throw new Error('Use a PNG, JPEG or WebP picture');
  }
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  try {
    const alpha = file.type !== 'image/jpeg';
    const max = alpha ? MAX_SIDE_WITH_ALPHA : MAX_SIDE;
    const scale = Math.min(1, max / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const context = canvas.getContext('2d');
    if (!context) throw new Error("This browser can't read pictures");
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL(alpha ? 'image/png' : 'image/jpeg', 0.92);
  } finally {
    bitmap.close();
  }
}
