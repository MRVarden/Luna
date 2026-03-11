// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import { Shield, ShieldCheck, ShieldAlert } from 'lucide-react'
import type { IdentitySnapshot, EpisodicSnapshot } from '../../api/types'

interface Props {
  identity: IdentitySnapshot | null
  episodic?: EpisodicSnapshot | null
}

export function IdentityPanel({ identity, episodic }: Props) {
  if (!identity) {
    return (
      <div className="flex items-center justify-center h-full text-luna-text-muted text-sm">
        Identité non chargée
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Integrity status */}
      <div className="flex items-center gap-3">
        <motion.div
          animate={identity.integrity_ok
            ? { scale: [1, 1.05, 1] }
            : { rotate: [0, 5, -5, 0] }
          }
          transition={{ duration: 3, repeat: Infinity }}
        >
          {identity.integrity_ok ? (
            <ShieldCheck className="w-6 h-6 text-emerald-400" />
          ) : (
            <ShieldAlert className="w-6 h-6 text-luna-accent" />
          )}
        </motion.div>
        <div>
          <div className={`text-sm font-medium ${identity.integrity_ok ? 'text-emerald-400' : 'text-luna-accent'}`}>
            {identity.integrity_ok ? 'Intégrité vérifiée' : 'Intégrité compromise'}
          </div>
          <div className="text-[10px] text-luna-text-muted font-mono">
            {identity.bundle_hash}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="flex flex-col items-center glass-panel px-2 py-2">
          <span className="font-mono text-lg text-luna-gold">{identity.kappa.toFixed(3)}</span>
          <span className="text-[8px] text-luna-text-muted uppercase tracking-wider">κ ancrage</span>
        </div>
        <div className="flex flex-col items-center glass-panel px-2 py-2">
          <span className="font-mono text-lg text-luna-primary">{identity.axioms_count}</span>
          <span className="text-[8px] text-luna-text-muted uppercase tracking-wider">Axiomes</span>
        </div>
        <div className="flex flex-col items-center glass-panel px-2 py-2">
          <span className="font-mono text-lg text-luna-cyan">Art.12</span>
          <span className="text-[8px] text-luna-text-muted uppercase tracking-wider">Protégé</span>
        </div>
      </div>

      {/* Episodic Memory */}
      {episodic && (
        <div className="flex items-center gap-3 glass-panel px-3 py-2">
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <span className="text-[9px] text-luna-text-muted uppercase tracking-wider">Mémoire épisodique</span>
              <span className="font-mono text-xs text-luna-text">
                {episodic.total_episodes} épisodes
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1 text-[10px]">
              <span className="text-luna-gold">{episodic.pinned_count} ancrés</span>
              {episodic.last_episode_outcome && (
                <span className={
                  episodic.last_episode_outcome === 'success' ? 'text-emerald-400' :
                  episodic.last_episode_outcome === 'veto' ? 'text-luna-accent' :
                  'text-luna-text-dim'
                }>
                  dernier: {episodic.last_episode_outcome}
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
