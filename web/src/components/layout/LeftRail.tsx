"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useHealth } from "@/lib/useHealth";

const MODES = [
  { href: "/retrieve", label: "Retrieve" },
  { href: "/evaluate", label: "Evaluate" },
  { href: "/hypothesize", label: "Hypothesize" },
] as const;

function Dot({ ok }: { ok: boolean | null }) {
  const color =
    ok === null ? "bg-ink-muted" : ok ? "bg-success" : "bg-danger";
  return <span className={`inline-block h-1.5 w-1.5 rounded-full ${color}`} />;
}

export function LeftRail() {
  const pathname = usePathname();
  const { health, error } = useHealth();

  const dbOk = health ? Boolean(health.db?.ok) : null;
  const redisOk = health ? Boolean(health.redis?.ok) : null;

  return (
    <aside className="flex w-52 shrink-0 flex-col border-r border-border bg-card">
      <nav className="flex flex-col gap-0.5 p-3">
        {MODES.map((m) => {
          const active = pathname === m.href || pathname.startsWith(`${m.href}/`);
          return (
            <Link
              key={m.href}
              href={m.href}
              className={`rounded border px-2.5 py-2 text-sm font-medium ${
                active
                  ? "border-accent bg-accent-soft text-accent"
                  : "border-transparent text-ink-secondary hover:bg-inset"
              }`}
            >
              {m.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto border-t border-border p-3">
        <div className="mb-2 text-xs font-medium text-ink-muted">Live status</div>
        <ul className="space-y-1.5 text-xs text-ink-secondary">
          <li className="flex items-center justify-between gap-2">
            <span>api</span>
            <span className="flex items-center gap-1.5 font-mono">
              <Dot ok={error ? false : health ? health.status === "ok" : null} />
              {error ? "down" : health?.status || "…"}
            </span>
          </li>
          <li className="flex items-center justify-between gap-2">
            <span>postgres</span>
            <span className="flex items-center gap-1.5 font-mono">
              <Dot ok={dbOk} />
              {dbOk == null ? "…" : dbOk ? "ok" : "err"}
            </span>
          </li>
          <li className="flex items-center justify-between gap-2">
            <span>redis</span>
            <span className="flex items-center gap-1.5 font-mono">
              <Dot ok={redisOk} />
              {redisOk == null ? "…" : redisOk ? "ok" : "err"}
            </span>
          </li>
          <li className="flex items-center justify-between gap-2">
            <span>ncbi</span>
            <span className="font-mono text-ink-muted">via api</span>
          </li>
          <li className="flex items-center justify-between gap-2">
            <span>medcpt</span>
            <span className="font-mono text-ink-muted">on demand</span>
          </li>
          <li className="flex items-center justify-between gap-2">
            <span>worker</span>
            <span className="font-mono text-ink-muted">arq</span>
          </li>
          <li className="flex items-center justify-between gap-2">
            <span>langfuse</span>
            <span className="font-mono text-ink-muted">optional</span>
          </li>
        </ul>
      </div>
    </aside>
  );
}
