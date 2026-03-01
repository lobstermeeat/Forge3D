import sharp from 'sharp';
import { getStorage } from './storage';

/**
 * Process and store a thumbnail from a base64 data URL.
 * Generates two sizes: full (1280x720) and preview (320x180).
 */
export async function processThumbnail(
  base64DataUrl: string,
  experienceId: string,
): Promise<{ thumbnailUrl: string; previewUrl: string }> {
  const storage = getStorage();

  // Strip data URL prefix: "data:image/...;base64,"
  const base64Data = base64DataUrl.replace(/^data:image\/\w+;base64,/, '');
  const buffer = Buffer.from(base64Data, 'base64');

  // Full size thumbnail
  const fullBuffer = await sharp(buffer)
    .resize(1280, 720, { fit: 'cover' })
    .jpeg({ quality: 85 })
    .toBuffer();

  const thumbnailUrl = await storage.write(
    `experiences/${experienceId}/thumb.jpg`,
    fullBuffer,
    'image/jpeg',
  );

  // Preview (small) thumbnail
  const previewBuffer = await sharp(buffer)
    .resize(320, 180, { fit: 'cover' })
    .jpeg({ quality: 70 })
    .toBuffer();

  const previewUrl = await storage.write(
    `experiences/${experienceId}/preview.jpg`,
    previewBuffer,
    'image/jpeg',
  );

  return { thumbnailUrl, previewUrl };
}
