// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import type { AffectSnapshot } from '../../api/types'

interface Props {
  affect: AffectSnapshot
}

function valenceColor(v: number): string {
  if (v > 0.3) return '#10b981'  // positive
  if (v < -0.3) return '#ef4444' // negative
  return '#53a8b6'               // neutral
}

function arousalHeight(a: number): string {
  return `${Math.max(a * 100, 8)}%`
}

export function AffectPanel({ affect }: Props) {
  const { affect: state, mood, emotions, uncovered } = affect

  return (
    <div className="flex flex-col gap-4">
      {/* VAD indicators */}
      <div className="flex gap-3">
        {[
          { label: 'Valence', value: state.valence, min: -1, max: 1, color: valenceColor(state.valence) },
          { label: 'Arousal', value: state.arousal, min: 0, max: 1, color: '#f5a623' },
          { label: 'Dominance', value: state.dominance, min: 0, max: 1, color: '#7c5cbf' },
        ].map(({ label, value, min, max, color }) => {
          const pct = ((value - min) / (max - min)) * 100
          return (
            <div key={label} className="flex-1 flex flex-col gap-1.5">
              <div className="flex justify-between">
                <span className="text-[10px] text-luna-text-dim uppercase tracking-wider">{label}</span>
                <span className="text-[11px] font-mono text-luna-text">{value.toFixed(2)}</span>
              </div>
              <div className="h-2 rounded-full bg-luna-surface-3 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: color }}
                  initial={false}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 1, ease: 'easeInOut' }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Emotions */}
      <div className="flex flex-wrap gap-2">
        {emotions.map((em, i) => (
          <motion.div
            key={em.fr}
            className="glass-panel flex items-center gap-2 px-3 py-1.5"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.1 }}
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{ backgroundColor: familyColor(em.family) }}
            />
            <span className="text-xs text-luna-text">{em.fr}</span>
            <span className="text-[10px] text-luna-text-muted italic">{em.en}</span>
            <span className="text-[10px] font-mono text-luna-text-dim">
              {(em.weight * 100).toFixed(0)}%
            </span>
          </motion.div>
        ))}
        {uncovered && (
          <motion.div
            className="glass-panel flex items-center gap-2 px-3 py-1.5 border-luna-accent/30"
            animate={{ borderColor: ['rgba(233,69,96,0.3)', 'rgba(233,69,96,0.6)', 'rgba(233,69,96,0.3)'] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            <span className="text-xs text-luna-accent">zone innommée</span>
            <span className="text-[10px] text-luna-accent/60">unnamed zone</span>
          </motion.div>
        )}
      </div>

      {/* Mood background */}
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-luna-text-muted uppercase tracking-wider">Mood</span>
        <div className="flex-1 flex gap-2">
          {['V', 'A', 'D'].map((dim, i) => {
            const vals = [mood.valence, mood.arousal, mood.dominance]
            return (
              <div key={dim} className="flex items-center gap-1">
                <span className="text-[9px] text-luna-text-muted">{dim}</span>
                <span className="text-[11px] font-mono text-luna-text-dim">
                  {vals[i].toFixed(2)}
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function familyColor(family: string): string {
  const colors: Record<string, string> = {
    joy: '#10b981',
    trust: '#53a8b6',
    anticipation: '#f5a623',
    surprise: '#a855f7',
    sadness: '#6366f1',
    fear: '#ef4444',
    anger: '#dc2626',
    complex: '#7c5cbf',
  }
  return colors[family] ?? '#7c5cbf'
}
