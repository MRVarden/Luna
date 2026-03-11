// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'

interface Props {
  psi: [number, number, number, number]
  psi0: [number, number, number, number]
}

const LABELS = ['Perception', 'Réflexion', 'Intégration', 'Expression']
const LABEL_ICONS = ['ψ₁', 'ψ₂', 'ψ₃', 'ψ₄']
const SIZE = 340
const CENTER = SIZE / 2
const RADIUS = 110

function polarToCart(angle: number, r: number): [number, number] {
  const rad = (angle - 90) * (Math.PI / 180)
  return [CENTER + r * Math.cos(rad), CENTER + r * Math.sin(rad)]
}

function makePolygon(values: number[], maxVal: number): string {
  return values
    .map((v, i) => {
      const angle = (360 / values.length) * i
      const r = (v / maxVal) * RADIUS
      const [x, y] = polarToCart(angle, r)
      return `${x},${y}`
    })
    .join(' ')
}

export function PsiRadar({ psi, psi0 }: Props) {
  const maxVal = 0.6 // Psi max practical value
  const rings = [0.15, 0.3, 0.45, 0.6]

  return (
    <div className="relative flex items-center justify-center">
      <svg
        width={SIZE}
        height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="consciousness-breathing"
      >
        <defs>
          <radialGradient id="psi-fill" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#7c5cbf" stopOpacity="0.4" />
            <stop offset="100%" stopColor="#533483" stopOpacity="0.1" />
          </radialGradient>
          <radialGradient id="psi0-fill" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#53a8b6" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#53a8b6" stopOpacity="0.03" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Grid rings */}
        {rings.map((r, i) => (
          <polygon
            key={i}
            points={makePolygon([r, r, r, r], maxVal)}
            fill="none"
            stroke="#2a2a5e"
            strokeWidth={i === rings.length - 1 ? 1 : 0.5}
            opacity={0.5}
          />
        ))}

        {/* Axis lines */}
        {[0, 1, 2, 3].map(i => {
          const angle = (360 / 4) * i
          const [x, y] = polarToCart(angle, RADIUS + 8)
          return (
            <line
              key={i}
              x1={CENTER} y1={CENTER}
              x2={x} y2={y}
              stroke="#2a2a5e"
              strokeWidth={0.5}
            />
          )
        })}

        {/* Psi0 anchor (identity) */}
        <motion.polygon
          points={makePolygon([...psi0], maxVal)}
          fill="url(#psi0-fill)"
          stroke="#53a8b6"
          strokeWidth={1}
          strokeDasharray="4 4"
          opacity={0.6}
          initial={false}
          animate={{ points: makePolygon([...psi0], maxVal) }}
          transition={{ duration: 1, ease: 'easeInOut' }}
        />

        {/* Psi current (consciousness) */}
        <motion.polygon
          points={makePolygon([...psi], maxVal)}
          fill="url(#psi-fill)"
          stroke="#7c5cbf"
          strokeWidth={2}
          filter="url(#glow)"
          initial={false}
          animate={{ points: makePolygon([...psi], maxVal) }}
          transition={{ duration: 1.5, ease: 'easeInOut' }}
        />

        {/* Psi vertex dots */}
        {psi.map((v, i) => {
          const angle = (360 / 4) * i
          const [x, y] = polarToCart(angle, (v / maxVal) * RADIUS)
          return (
            <motion.circle
              key={i}
              r={4}
              fill="#7c5cbf"
              stroke="#e8e8f0"
              strokeWidth={1.5}
              initial={false}
              animate={{ cx: x, cy: y }}
              transition={{ duration: 1.5, ease: 'easeInOut' }}
            />
          )
        })}

        {/* Labels */}
        {LABELS.map((label, i) => {
          const angle = (360 / 4) * i
          const [x, y] = polarToCart(angle, RADIUS + 30)
          return (
            <g key={i}>
              <text
                x={x} y={y - 6}
                textAnchor="middle"
                className="fill-luna-text-dim text-[10px]"
              >
                {label}
              </text>
              <text
                x={x} y={y + 8}
                textAnchor="middle"
                className="fill-luna-primary text-[11px] font-mono font-semibold"
              >
                {LABEL_ICONS[i]} {psi[i].toFixed(3)}
              </text>
            </g>
          )
        })}

        {/* Center phi symbol */}
        <text
          x={CENTER} y={CENTER + 1}
          textAnchor="middle"
          dominantBaseline="central"
          className="fill-luna-primary/30 text-[24px] font-light"
        >
          Ψ
        </text>
      </svg>
    </div>
  )
}
