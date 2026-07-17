import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        page: "var(--surface-page)",
        card: "var(--surface-card)",
        inset: "var(--surface-inset)",
        border: "var(--border)",
        ink: {
          DEFAULT: "var(--ink)",
          secondary: "var(--ink-secondary)",
          muted: "var(--ink-muted)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          soft: "var(--accent-soft)",
          fill: "var(--accent-fill)",
          on: "var(--accent-on)",
          border: "var(--accent-border)",
        },
        success: {
          DEFAULT: "var(--success)",
          fill: "var(--success-fill)",
          bg: "var(--success-bg)",
          border: "var(--success-border)",
        },
        danger: {
          DEFAULT: "var(--danger)",
          fill: "var(--danger-fill)",
          bg: "var(--danger-bg)",
          border: "var(--danger-border)",
        },
        warning: {
          DEFAULT: "var(--warning)",
          fill: "var(--warning-fill)",
          bg: "var(--warning-bg)",
          border: "var(--warning-border)",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      borderRadius: {
        card: "var(--radius-card)",
      },
      fontWeight: {
        normal: "400",
        medium: "500",
      },
    },
  },
  plugins: [],
};

export default config;
