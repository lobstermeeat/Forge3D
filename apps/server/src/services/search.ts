import { MeiliSearch } from 'meilisearch';

const MEILI_URL = process.env['MEILI_URL'] ?? 'http://localhost:7700';
const MEILI_MASTER_KEY = process.env['MEILI_MASTER_KEY'] ?? 'forge3d_search_dev';

const client = new MeiliSearch({
  host: MEILI_URL,
  apiKey: MEILI_MASTER_KEY,
});

const INDEX_NAME = 'experiences';

let indexReady = false;

export async function ensureIndex(): Promise<void> {
  if (indexReady) return;
  try {
    const index = client.index(INDEX_NAME);
    await index.updateFilterableAttributes(['categoryId', 'formatType', 'creatorId']);
    await index.updateSortableAttributes(['publishedAt', 'viewCount', 'likeCount']);
    indexReady = true;
  } catch (err) {
    console.error('[Search] Failed to initialize index:', err);
  }
}

export interface SearchDocument {
  id: string;
  title: string;
  description: string | null;
  tags: string[];
  creatorId: string;
  creatorName: string;
  categoryId: string | null;
  formatType: string;
  publishedAt: string | null;
  viewCount: number;
  likeCount: number;
}

export async function indexExperience(doc: SearchDocument): Promise<void> {
  try {
    await ensureIndex();
    await client.index(INDEX_NAME).addDocuments([doc]);
  } catch (err) {
    console.error('[Search] Failed to index experience:', err);
  }
}

export async function removeExperience(id: string): Promise<void> {
  try {
    await client.index(INDEX_NAME).deleteDocument(id);
  } catch (err) {
    console.error('[Search] Failed to remove experience:', err);
  }
}

export interface SearchOptions {
  query: string;
  categoryId?: string;
  formatType?: string;
  sort?: 'relevance' | 'newest' | 'popular';
  limit?: number;
  offset?: number;
}

export interface SearchHit {
  id: string;
  title: string;
  description: string | null;
  creatorId: string;
  creatorName: string;
  categoryId: string | null;
  formatType: string;
  publishedAt: string | null;
  viewCount: number;
  likeCount: number;
}

export interface SearchResponse {
  hits: SearchHit[];
  totalHits: number;
}

export async function searchExperiences(opts: SearchOptions): Promise<SearchResponse> {
  try {
    await ensureIndex();

    const filter: string[] = [];
    if (opts.categoryId) filter.push(`categoryId = "${opts.categoryId}"`);
    if (opts.formatType) filter.push(`formatType = "${opts.formatType}"`);

    let sort: string[] | undefined;
    if (opts.sort === 'newest') sort = ['publishedAt:desc'];
    else if (opts.sort === 'popular') sort = ['likeCount:desc', 'viewCount:desc'];

    const results = await client.index(INDEX_NAME).search<SearchHit>(opts.query, {
      filter: filter.length > 0 ? filter : undefined,
      sort,
      limit: opts.limit ?? 20,
      offset: opts.offset ?? 0,
    });

    return {
      hits: results.hits,
      totalHits: typeof results.estimatedTotalHits === 'number' ? results.estimatedTotalHits : results.hits.length,
    };
  } catch {
    // Meilisearch unavailable — return empty
    return { hits: [], totalHits: 0 };
  }
}
