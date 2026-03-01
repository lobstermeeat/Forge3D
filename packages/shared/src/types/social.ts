export interface PaginatedFeed<T> {
  items: T[];
  nextCursor: string | null;
  hasMore: boolean;
}
