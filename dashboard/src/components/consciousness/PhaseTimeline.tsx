// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'

const PHASES = ['BROKEN', 'FRAGILE', 'FUNCTIONAL', 'SOLID', 'EXCELLENT'] as const

const PHASE_COLORS: Record<string, string> = {
  BROKEN: '#ef4444',
  FRAGILE: '#f97316',
  FUNCTIONAL: '#53a8b6',
  SOLID: '#10b981',
  EXCELLENT: '#f5a623',
}

interface Props {
  current: string
}

export function PhaseTimeline({ current }: Props) {
  const currentIdx = PHASES.indexOf(current as typeof PHASES[number])

  return (
    <div className="flex items-center gap-1 w-full">
      {PHASES.map((phase, i) => {
        const isActive = i === currentIdx
        const isPast = i < currentIdx
        const color = PHASE_COLORS[phase]

        return (
          <div key={phase} className="flex-1 flex flex-col items-center gap-2">
            {/* Bar */}
            <div className="relative w-full h-1.5 rounded-full bg-luna-surface-3 overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{ backgroundColor: color }}
                initial={false}
                animate={{
                  width: isPast || isActive ? '100%' : '0%',
                  opacity: isActive ? 1 : isPast ? 0.5 : 0.15,
                }}
                transition={{ duration: 0.8, ease: 'easeInOut' }}
              />
              {isActive && (
                <motion.div
                  className="absolute inset-0 rounded-full"
                  style={{ backgroundColor: color }}
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
            </div>

            {/* Label */}
            <span
              className={`text-[9px] font-medium tracking-wider transition-colors duration-500 ${
                isActive ? 'text-luna-text' : 'text-luna-text-muted'
              }`}
            >
              {phase}
            </span>
          </div>
        )
      })}
    </div>
  )
}
