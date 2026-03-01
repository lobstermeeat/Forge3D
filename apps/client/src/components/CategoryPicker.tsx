import { trpc } from '@/api/trpc';

interface CategoryPickerProps {
  selected: string | null;
  onSelect: (categoryId: string | null) => void;
}

export function CategoryPicker({ selected, onSelect }: CategoryPickerProps) {
  const categoriesQuery = trpc.category.list.useQuery();
  const categories = categoriesQuery.data ?? [];

  return (
    <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-none">
      <button
        onClick={() => onSelect(null)}
        className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
          selected === null
            ? 'bg-[#89b4fa] text-[#11111b]'
            : 'bg-[#313244] text-[#a6adc8] hover:bg-[#45475a] hover:text-[#cdd6f4]'
        }`}
      >
        All
      </button>
      {categories.map((cat) => (
        <button
          key={cat.id}
          onClick={() => onSelect(cat.id)}
          className={`shrink-0 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
            selected === cat.id
              ? 'bg-[#89b4fa] text-[#11111b]'
              : 'bg-[#313244] text-[#a6adc8] hover:bg-[#45475a] hover:text-[#cdd6f4]'
          }`}
        >
          {cat.name}
        </button>
      ))}
    </div>
  );
}
