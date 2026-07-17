"use client";

type Chip = {
  id: string;
  label: string;
};

type Props = {
  chips: Chip[];
  selected: string[];
  onToggle: (id: string) => void;
  multi?: boolean;
};

export function FilterChips({ chips, selected, onToggle, multi = true }: Props) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {chips.map((chip) => {
        const on = selected.includes(chip.id);
        return (
          <button
            key={chip.id}
            type="button"
            onClick={() => {
              if (!multi && on) return;
              onToggle(chip.id);
            }}
            className={`rounded border px-2 py-1 text-xs font-medium ${
              on
                ? "border-accent bg-accent-soft text-accent"
                : "border-border bg-card text-ink-secondary hover:bg-inset"
            }`}
          >
            {chip.label}
          </button>
        );
      })}
    </div>
  );
}
