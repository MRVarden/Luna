// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import type { CycleRecord, CausalGraphSnapshot } from '../../api/types'

interface Props {
  lastCycle: CycleRecord | null
  psi?: [number, number, number, number] | null
  causalGraph?: CausalGraphSnapshot | null
}

// v5.0: 4 cognitive stages (Luna's own faculties, not external agents)
const STAGES = [
  { id: 'perception', label: 'Perception', sub: 'ψ₁ Observer', color: '#e94560' },
  { id: 'reflection', label: 'Réflexion', sub: 'ψ₂ Analyser', color: '#7c5cbf' },
  { id: 'integration', label: 'Intégration', sub: 'ψ₃ Synthétiser', color: '#53a8b6' },
  { id: 'expression', label: 'Expression', sub: 'ψ₄ Communiquer', color: '#f5a623' },
]

export function CognitiveFlow({ lastCycle, psi, causalGraph }: Props) {
  // Dominant Ψ component = active cognitive stage
  const activeStage = psi
    ? psi.indexOf(Math.max(...psi))
    : -1

  return (
    <div className="flex flex-col gap-4">
      {/* Cognitive stage nodes */}
      <div className="flex items-center justify-between gap-2">
        {STAGES.map((stage, i) => {
          const isActive = activeStage === i

          return (
            <div key={stage.id} className="flex items-center gap-2 flex-1">
              {/* Node */}
              <motion.div
                className="flex-1 glass-panel px-3 py-2.5 text-center"
                animate={{
                  borderColor: isActive
                    ? [stage.color + '30', stage.color + '80', stage.color + '30']
                    : stage.color + '15',
                  opacity: isActive ? 1 : 0.45,
                }}
                transition={isActive ? { duration: 2, repeat: Infinity } : { duration: 0.5 }}
              >
                <div className="text-xs font-semibold" style={{ color: stage.color }}>
                  {stage.label}
                </div>
                <div className="text-[9px] text-luna-text-muted mt-0.5">{stage.sub}</div>
                {psi && (
                  <div className="text-[10px] font-mono mt-1" style={{ color: isActive ? stage.color : '#666' }}>
                    {psi[i].toFixed(3)}
                  </div>
                )}
              </motion.div>

              {/* Arrow */}
              {i < STAGES.length - 1 && (
                <motion.div
                  className="flex-shrink-0"
                  animate={{
                    opacity: isActive ? 0.8 : 0.2,
                  }}
                  transition={{ duration: 0.5 }}
                >
                  <svg width="20" height="12" viewBox="0 0 20 12">
                    <path
                      d="M0 6 L14 6 M10 2 L16 6 L10 10"
                      fill="none"
                      stroke={isActive ? stage.color : '#533483'}
                      strokeWidth="1.5"
                    />
                  </svg>
                </motion.div>
              )}
            </div>
          )
        })}
      </div>

      {/* Last cycle info */}
      {lastCycle && (
        <div className="flex items-center gap-4 text-[10px]">
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${
              lastCycle.intent === 'DREAM' ? 'bg-luna-primary' :
              lastCycle.intent === 'ALERT' ? 'bg-red-400' :
              lastCycle.intent === 'INTROSPECT' ? 'bg-purple-400' :
              'bg-luna-cyan'
            }`} />
            <span className="text-luna-text-dim">Intent:</span>
            <span className="font-mono text-luna-text">{lastCycle.intent}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-luna-text-dim">Confiance:</span>
            <span className="font-mono text-luna-text">{lastCycle.thinker_confidence.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-luna-text-dim">Obs:</span>
            <span className="font-mono text-luna-text">{lastCycle.observations.length}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-luna-text-dim">Duree:</span>
            <span className="font-mono text-luna-text">{lastCycle.duration_seconds.toFixed(1)}s</span>
          </div>
        </div>
      )}

      {/* Causal Graph stats */}
      {causalGraph && (
        <div className="flex items-center gap-4 text-[10px] text-luna-text-dim mt-2">
          <span>Graphe causal: <span className="font-mono text-luna-cyan">{causalGraph.node_count}</span> nœuds</span>
          <span><span className="font-mono text-luna-primary">{causalGraph.edge_count}</span> arêtes</span>
          <span><span className="font-mono text-emerald-400">{causalGraph.confirmed_count}</span> confirmées</span>
          {causalGraph.density > 0 && (
            <span>densité: <span className="font-mono text-luna-gold">{causalGraph.density.toFixed(3)}</span></span>
          )}
        </div>
      )}
    </div>
  )
}
