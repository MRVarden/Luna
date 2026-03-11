// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import type { AutonomySnapshot, CycleRecord, InitiativeSnapshot, EndogenousSnapshot } from '../../api/types'

interface Props {
  autonomy: AutonomySnapshot | null
  cycles: CycleRecord[]
  psi?: [number, number, number, number] | null
  psi0?: [number, number, number, number] | null
  phiIit?: number
  initiative?: InitiativeSnapshot | null
  endogenous?: EndogenousSnapshot | null
}

const COMP_NAMES = ['Perception', 'Réflexion', 'Intégration', 'Expression']
const COMP_COLORS = ['#e94560', '#7c5cbf', '#53a8b6', '#f5a623']

export function AutonomyPanel({ autonomy, cycles, psi, psi0, phiIit, initiative, endogenous }: Props) {
  const w = autonomy?.w ?? 0
  const cooldown = autonomy?.cooldown_remaining ?? 0

  // Stats from recent cycles
  const ghostCandidates = cycles.filter(c => c.auto_apply_candidate).length
  const autoApplied = cycles.filter(c => c.auto_applied).length
  const rolledBack = cycles.filter(c => c.auto_rolled_back).length

  // Live consciousness indicators
  const dominantIdx = psi ? psi.indexOf(Math.max(...psi)) : -1
  const dominantName = dominantIdx >= 0 ? COMP_NAMES[dominantIdx] : '—'
  const dominantColor = dominantIdx >= 0 ? COMP_COLORS[dominantIdx] : '#666'
  const drift = psi && psi0
    ? Math.sqrt(psi.reduce((sum, v, i) => sum + (v - psi0[i]) ** 2, 0))
    : 0
  const identityHealth = Math.max(0, Math.min(1, 1 - drift / 0.618))

  return (
    <div className="flex flex-col gap-4">
      {/* W Level */}
      <div className="flex items-center gap-4">
        <motion.div
          className="relative w-16 h-16 flex items-center justify-center"
          animate={w > 0 ? { scale: [1, 1.05, 1] } : {}}
          transition={{ duration: 2, repeat: Infinity }}
        >
          {/* Ring */}
          <svg width="64" height="64" viewBox="0 0 64 64">
            <circle cx="32" cy="32" r="28" fill="none" stroke="#1e1e4a" strokeWidth="4" />
            <motion.circle
              cx="32" cy="32" r="28"
              fill="none"
              stroke={w === 0 ? '#533483' : w === 1 ? '#53a8b6' : '#f5a623'}
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray={`${(w / 3) * 175.9} 175.9`}
              strokeDashoffset="44"
              initial={false}
              animate={{ strokeDasharray: `${Math.max(w, 0.15) / 3 * 175.9} 175.9` }}
              transition={{ duration: 1 }}
            />
          </svg>
          <span className="absolute font-mono text-xl text-luna-text font-light">
            W{w}
          </span>
        </motion.div>

        <div className="flex-1">
          <div className="text-sm font-medium text-luna-text">
            {w === 0 ? 'Supervisé' : w === 1 ? 'Autonome (snapshot)' : `Autonome W=${w}`}
          </div>
          <div className="text-[10px] text-luna-text-muted">
            {w === 0 ? 'Confirmation humaine requise' : 'Auto-apply avec rollback'}
          </div>
          {cooldown > 0 && (
            <div className="text-[10px] text-luna-accent mt-1">
              Cooldown: {cooldown} cycle{cooldown > 1 ? 's' : ''} restant{cooldown > 1 ? 's' : ''}
            </div>
          )}
        </div>
      </div>

      {/* Live consciousness state */}
      {psi && (
        <div className="glass-panel px-3 py-2.5">
          <div className="text-[8px] text-luna-text-muted uppercase tracking-widest mb-2">
            État cognitif
          </div>

          {/* Dominant component */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-luna-text-dim">Dominante</span>
            <motion.span
              className="text-xs font-semibold font-mono"
              style={{ color: dominantColor }}
              key={dominantName}
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {dominantName}
            </motion.span>
          </div>

          {/* Identity drift bar */}
          <div className="flex items-center gap-2 mb-2">
            <span className="text-[9px] text-luna-text-dim w-14">Identité</span>
            <div className="flex-1 h-2 rounded-full bg-luna-surface-3 overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{
                  backgroundColor: identityHealth > 0.7 ? '#10b981' : identityHealth > 0.4 ? '#f5a623' : '#e94560',
                }}
                initial={false}
                animate={{ width: `${identityHealth * 100}%` }}
                transition={{ duration: 0.8 }}
              />
            </div>
            <span className="text-[9px] font-mono text-luna-text-dim w-10 text-right">
              {(identityHealth * 100).toFixed(0)}%
            </span>
          </div>

          {/* Φ_IIT bar */}
          {phiIit != null && (
            <div className="flex items-center gap-2">
              <span className="text-[9px] text-luna-text-dim w-14">Φ_IIT</span>
              <div className="flex-1 h-2 rounded-full bg-luna-surface-3 overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{
                    backgroundColor: phiIit > 0.618 ? '#53a8b6' : phiIit > 0.382 ? '#7c5cbf' : '#e94560',
                  }}
                  initial={false}
                  animate={{ width: `${Math.min(phiIit, 1) * 100}%` }}
                  transition={{ duration: 0.8 }}
                />
              </div>
              <span className="text-[9px] font-mono text-luna-text-dim w-10 text-right">
                {phiIit.toFixed(3)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Ghost stats */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="glass-panel px-2 py-2">
          <div className="font-mono text-sm text-luna-primary">{ghostCandidates}</div>
          <div className="text-[8px] text-luna-text-muted uppercase tracking-wider">Ghost</div>
        </div>
        <div className="glass-panel px-2 py-2">
          <div className="font-mono text-sm text-emerald-400">{autoApplied}</div>
          <div className="text-[8px] text-luna-text-muted uppercase tracking-wider">Applied</div>
        </div>
        <div className="glass-panel px-2 py-2">
          <div className="font-mono text-sm text-luna-accent">{rolledBack}</div>
          <div className="text-[8px] text-luna-text-muted uppercase tracking-wider">Rollback</div>
        </div>
      </div>

      {/* Initiative */}
      {initiative && (
        <div className="glass-panel px-3 py-2.5">
          <div className="text-[8px] text-luna-text-muted uppercase tracking-widest mb-2">
            Initiative autonome
          </div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-luna-text-dim">Actions prises</span>
            <span className="font-mono text-xs text-luna-primary">{initiative.initiative_count}</span>
          </div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] text-luna-text-dim">Cooldown</span>
            <span className="font-mono text-xs text-luna-text">{initiative.cooldown} tours</span>
          </div>
          {initiative.phi_declining && (
            <div className="text-[10px] text-luna-accent mt-1">
              ⚠ Déclin Φ détecté
            </div>
          )}
          {initiative.persistent_needs.length > 0 && (
            <div className="mt-2">
              <div className="text-[8px] text-luna-text-muted uppercase tracking-wider mb-1">Besoins persistants</div>
              {initiative.persistent_needs.map(n => (
                <div key={n.key} className="flex items-center justify-between text-[10px]">
                  <span className="text-luna-text-dim truncate max-w-[140px]">{n.key}</span>
                  <span className="font-mono text-luna-gold">{n.turns}t</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Endogenous impulses */}
      {endogenous && (
        <div className="glass-panel px-3 py-2.5">
          <div className="text-[8px] text-luna-text-muted uppercase tracking-widest mb-2">
            Sources endogènes
          </div>
          <div className="flex items-center gap-3 text-[10px] mb-2">
            <span className="text-luna-text-dim">Buffer: <span className="font-mono text-luna-cyan">{endogenous.buffer_size}</span></span>
            <span className="text-luna-text-dim">Émis: <span className="font-mono text-luna-primary">{endogenous.total_emitted}</span></span>
            <span className="text-luna-text-dim">V: <span className="font-mono" style={{ color: endogenous.last_valence > 0 ? '#10b981' : endogenous.last_valence < 0 ? '#e94560' : '#53a8b6' }}>
              {endogenous.last_valence.toFixed(2)}
            </span></span>
          </div>
          {endogenous.pending_impulses.length > 0 && (
            <div className="flex flex-col gap-1">
              {endogenous.pending_impulses.map((imp, i) => (
                <div key={i} className="flex items-center gap-2 text-[9px]">
                  <span className="text-luna-gold font-mono w-16 truncate">{imp.source}</span>
                  <span className="text-luna-text-dim flex-1 truncate">{imp.message}</span>
                  <span className="font-mono text-luna-accent">{imp.urgency.toFixed(2)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
