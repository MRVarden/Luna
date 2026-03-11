/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        luna: {
          bg: '#08081a',
          surface: '#0e0e24',
          'surface-2': '#161638',
          'surface-3': '#1e1e4a',
          border: '#2a2a5e',
          'border-glow': '#533483',
          primary: '#7c5cbf',
          'primary-dim': '#533483',
          accent: '#e94560',
          cyan: '#53a8b6',
          gold: '#f5a623',
          text: '#e8e8f0',
          'text-dim': '#7878a0',
          'text-muted': '#4a4a70',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'luna': '0 0 30px rgba(83, 52, 131, 0.15)',
        'luna-glow': '0 0 40px rgba(83, 52, 131, 0.3)',
        'accent-glow': '0 0 30px rgba(233, 69, 96, 0.2)',
        'cyan-glow': '0 0 30px rgba(83, 168, 182, 0.2)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'breathe': 'breathe 6s ease-in-out infinite',
        'glow': 'glow 3s ease-in-out infinite alternate',
      },
      keyframes: {
        breathe: {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.8' },
        },
        glow: {
          '0%': { boxShadow: '0 0 20px rgba(83, 52, 131, 0.1)' },
          '100%': { boxShadow: '0 0 40px rgba(83, 52, 131, 0.3)' },
        },
      },
    },
  },
  plugins: [],
}
