import { create } from 'zustand';

export interface ImportedAsset {
  id: string;
  name: string;
  file: File;
}

interface AssetState {
  imports: ImportedAsset[];
  addImport: (asset: ImportedAsset) => void;
}

/** Models imported this session, so they can be placed again from the Library. */
export const useAssetStore = create<AssetState>((set) => ({
  imports: [],
  addImport: (asset) =>
    set((s) => ({ imports: [asset, ...s.imports.filter((a) => a.name !== asset.name)] })),
}));
