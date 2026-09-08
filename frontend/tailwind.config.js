/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        soc: {
          bg: "#06090e",
          surface: "#0b1018",
          panel: "#101725",
          border: "#1c273b",
          cyan: "#00f0ff",
          blue: "#38bdf8",
          glow: "#0284c7",
          amber: "#f59e0b",
          red: "#ef4444",
          crimson: "#dc2626",
          emerald: "#10b981",
          muted: "#64748b"
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar-sweep': 'radarSweep 4s linear infinite',
        'threat-flash': 'threatFlash 1s ease-in-out infinite',
      },
      keyframes: {
        radarSweep: {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },
        threatFlash: {
          '0%, 100%': { opacity: '1', borderColor: '#ef4444' },
          '50%': { opacity: '0.4', borderColor: '#7f1d1d' },
        }
      }
    },
  },
  plugins: [],
}
