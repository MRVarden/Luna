// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import type { RewardVector } from '../../api/types'

interface Props {
  reward: RewardVector | null    // last cycle's reward
  liveReward: RewardVector | null // real-time Evaluator output
}

// 9 cognitive components — matches Python Evaluator exactly
const COMPONENT_LABELS: Record<string, string> = {
  constitution_integrity: 'Constitution',
  anti_collapse: 'Anti-collapse',
  integration_coherence: 'Intégration',
  identity_stability: 'Identité',
  reflection_depth: 'Réflexion',
  perception_acuity: 'Perception',
  expression_fidelity: 'Expression',
  affect_regulation: 'Affect',
  memory_vitality: 'Mémoire',
}

const COMPONENT_COLORS: Record<string, string> = {
  constitution_integrity: '#e94560',
  anti_collapse: '#ef4444',
  integration_coherence: '#53a8b6',
  identity_stability: '#7c5cbf',
  reflection_depth: '#a855f7',
  perception_acuity: '#f97316',
  expression_fidelity: '#f5a623',
  affect_regulation: '#ec4899',
  memory_vitality: '#10b981',
}

const COMPONENT_GROUPS: Record<string, string> = {
  constitution_integrity: 'Sécurité',
  integration_coherence: 'ψ₃',
  reflection_depth: 'ψ₂',
  perception_acuity: 'ψ₁',
  expression_fidelity: 'ψ₄',
  affect_regulation: 'Trans.',
}

export function RewardPanel({ reward, liveReward }: Props) {
  // Prefer live (real-time Evaluator), fallback to last cycle reward
  const active = liveReward ?? reward

  if (!active) {
    return (
      <div className="flex items-center justify-center h-full text-luna-text-muted text-sm">
        En attente du juge...
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Header stats */}
      <div className="flex items-center gap-4">
        {reward && (
          <div className="flex flex-col items-center">
            <span className="font-mono text-2xl text-luna-text font-light">
              #{reward.dominance_rank}
            </span>
            <span className="text-[9px] text-luna-text-muted uppercase tracking-wider">Rank</span>
          </div>
        )}
        {reward && (
          <div className="flex flex-col items-center">
            <span className={`font-mono text-xl font-light ${
              reward.delta_j >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}>
              {reward.delta_j >= 0 ? '+' : ''}{reward.delta_j.toFixed(4)}
            </span>
            <span className="text-[9px] text-luna-text-muted uppercase tracking-wider">ΔJ</span>
          </div>
        )}
        {liveReward && (
          <div className="ml-auto">
            <span className="text-[8px] text-luna-cyan uppercase tracking-widest">
              Live
            </span>
          </div>
        )}
      </div>

      {/* Component bars */}
      <div className="flex flex-col gap-1">
        {active.components.map((comp, i) => {
          const color = COMPONENT_COLORS[comp.name] ?? '#533483'
          const label = COMPONENT_LABELS[comp.name] ?? comp.name
          const group = COMPONENT_GROUPS[comp.name]

          return (
            <div key={comp.name}>
              {/* Group separator */}
              {group && (
                <div className="text-[7px] text-luna-text-muted uppercase tracking-widest mt-1 mb-0.5 pl-1">
                  {group}
                </div>
              )}
              <div className="flex items-center gap-2">
                <span className="text-[9px] text-luna-text-dim w-20 text-right truncate">
                  {label}
                </span>
                <div className="flex-1 h-3 rounded-full bg-luna-surface-3 overflow-hidden relative">
                  {/* Center line (0 value) */}
                  <div className="absolute left-1/2 top-0 bottom-0 w-px bg-luna-border" />
                  <motion.div
                    className="absolute top-0 bottom-0 rounded-full"
                    style={{ backgroundColor: color }}
                    initial={false}
                    animate={{
                      left: comp.value >= 0 ? '50%' : `${((comp.value + 1) / 2) * 100}%`,
                      width: `${Math.abs(comp.value) * 50}%`,
                    }}
                    transition={{ duration: 0.8, delay: i * 0.05 }}
                  />
                </div>
                <span className="text-[10px] font-mono text-luna-text-dim w-10 text-right">
                  {comp.value.toFixed(2)}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
