// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import { Moon, Sparkles } from 'lucide-react'
import type { DreamStatus } from '../../api/types'

interface Props {
  dream: DreamStatus | null
}

const PHASES = [
  'CONSOLIDATION', 'REINTERPRETATION', 'DEFRAGMENTATION',
  'CREATIVE', 'HARVEST', 'REPLAY', 'EXPLORATION', 'SIM_CONSOLIDATION',
]

export function DreamPanel({ dream }: Props) {
  if (!dream) {
    return (
      <div className="flex items-center justify-center h-full text-luna-text-muted text-sm">
        Système de rêve non connecté
      </div>
    )
  }

  const isSleeping = dream.state !== 'awake'

  return (
    <div className="flex flex-col gap-4">
      {/* Status */}
      <div className="flex items-center gap-3">
        <motion.div
          animate={isSleeping ? {
            rotate: [0, 10, -10, 0],
            scale: [1, 1.1, 1],
          } : {}}
          transition={isSleeping ? { duration: 3, repeat: Infinity } : {}}
        >
          {isSleeping ? (
            <Moon className="w-5 h-5 text-luna-primary" />
          ) : (
            <Sparkles className="w-5 h-5 text-luna-gold" />
          )}
        </motion.div>
        <div>
          <div className={`text-sm font-medium ${isSleeping ? 'text-luna-primary' : 'text-luna-text'}`}>
            {dream.state}
          </div>
          <div className="text-[10px] text-luna-text-muted">
            {dream.dream_count} rêves — {dream.total_dream_time.toFixed(1)}s total
          </div>
        </div>
      </div>

      {/* Dream phases (shown when sleeping) */}
      {isSleeping && (
        <div className="flex gap-1">
          {PHASES.map((phase, i) => (
            <motion.div
              key={phase}
              className="flex-1 h-1.5 rounded-full"
              style={{ backgroundColor: '#7c5cbf' }}
              initial={{ opacity: 0.15 }}
              animate={{
                opacity: [0.15, 0.8, 0.15],
              }}
              transition={{
                duration: 2,
                delay: i * 0.25,
                repeat: Infinity,
              }}
            />
          ))}
        </div>
      )}

      {/* Last dream */}
      {dream.last_dream_at && (
        <div className="flex items-center gap-4 text-[10px] text-luna-text-dim">
          <span>Dernier: {new Date(dream.last_dream_at).toLocaleTimeString('fr-FR')}</span>
          <span>{dream.last_dream_duration.toFixed(1)}s</span>
        </div>
      )}

      {/* Dream insights */}
      <div className="grid grid-cols-2 gap-2">
        {dream.skills_learned > 0 && (
          <div className="glass-panel px-2 py-1.5 flex flex-col items-center">
            <span className="font-mono text-sm text-luna-gold">{dream.skills_learned}</span>
            <span className="text-[8px] text-luna-text-muted uppercase tracking-wider">Skills</span>
          </div>
        )}
        {dream.dream_mode && (
          <div className="glass-panel px-2 py-1.5 flex flex-col items-center">
            <span className="font-mono text-xs text-luna-primary">{dream.dream_mode}</span>
            <span className="text-[8px] text-luna-text-muted uppercase tracking-wider">Mode</span>
          </div>
        )}
      </div>

      {/* Psi0 Drift */}
      {dream.psi0_drift && dream.psi0_drift.some(d => Math.abs(d) > 0.001) && (
        <div>
          <div className="text-[8px] text-luna-text-muted uppercase tracking-widest mb-1.5">
            Dérive Ψ₀
          </div>
          <div className="flex gap-1.5">
            {['P', 'R', 'I', 'E'].map((dim, i) => {
              const val = dream.psi0_drift[i] ?? 0
              const color = val > 0 ? '#10b981' : val < -0.001 ? '#e94560' : '#53a8b6'
              return (
                <div key={dim} className="flex-1 flex flex-col items-center gap-0.5">
                  <span className="text-[9px] text-luna-text-muted">{dim}</span>
                  <span className="text-[10px] font-mono" style={{ color }}>
                    {val >= 0 ? '+' : ''}{val.toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
