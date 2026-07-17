"use client";

import { useEffect, useState } from "react";
import { useHealth } from "@/lib/useHealth";

function applyTheme(isDark: boolean) {
  const root = document.documentElement;
  root.classList.toggle("dark", isDark);
  root.classList.toggle("light", !isDark);
  root.dataset.theme = isDark ? "dark" : "light";
}

export function TopBar() {
  const { health } = useHealth();
  const [dark, setDark] = useState(false);

  useEffect(() => {
    const stored = window.localStorage.getItem("biolit-theme");
    const preferDark =
      stored === "dark" ||
      (!stored && window.matchMedia("(prefers-color-scheme: dark)").matches);
    setDark(preferDark);
    applyTheme(preferDark);
  }, []);

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    applyTheme(next);
    window.localStorage.setItem("biolit-theme", next ? "dark" : "light");
  };

  const env = process.env.NEXT_PUBLIC_APP_ENV || "development";

  return (
    <header className="flex h-12 items-center justify-between border-b border-border bg-card px-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium tracking-tight text-ink">BioLit</span>
        <span className="rounded border border-border px-1.5 py-0.5 font-mono text-xs text-ink-muted">
          {env}
        </span>
      </div>
      <div className="flex items-center gap-4 text-xs text-ink-secondary">
        <span className="font-mono">
          cache {health?.redis?.ok ? "redis" : "—"}
        </span>
        <span className="font-mono">
          api {health?.status || "…"}
        </span>
        <button
          type="button"
          onClick={toggleTheme}
          className="rounded border border-border px-2 py-1 font-medium hover:bg-inset"
        >
          {dark ? "Light" : "Dark"}
        </button>
      </div>
    </header>
  );
}
