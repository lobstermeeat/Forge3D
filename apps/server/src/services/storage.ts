import fs from 'node:fs/promises';
import path from 'node:path';

export interface StorageProvider {
  write(key: string, data: Buffer, contentType: string): Promise<string>;
  read(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  getUrl(key: string): string;
}

/**
 * Local filesystem storage provider for development.
 * Files are stored in UPLOAD_DIR and served via @fastify/static.
 *
 * Swap this with S3StorageProvider when budget allows.
 */
export class LocalStorageProvider implements StorageProvider {
  constructor(
    private uploadDir: string,
    private baseUrl: string,
  ) {}

  async write(key: string, data: Buffer, _contentType: string): Promise<string> {
    const filePath = path.join(this.uploadDir, key);
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, data);
    return this.getUrl(key);
  }

  async read(key: string): Promise<Buffer> {
    const filePath = path.join(this.uploadDir, key);
    return fs.readFile(filePath);
  }

  async delete(key: string): Promise<void> {
    const filePath = path.join(this.uploadDir, key);
    await fs.unlink(filePath).catch(() => {});
  }

  getUrl(key: string): string {
    return `${this.baseUrl}/${key}`;
  }
}

// Singleton
let storageInstance: StorageProvider | null = null;

export function getStorage(): StorageProvider {
  if (!storageInstance) {
    const uploadDir = process.env['UPLOAD_DIR'] ?? './uploads';
    const baseUrl = process.env['STORAGE_BASE_URL'] ?? 'http://localhost:4000/uploads';
    storageInstance = new LocalStorageProvider(uploadDir, baseUrl);
  }
  return storageInstance;
}
