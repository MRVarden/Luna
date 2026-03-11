// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import { Activity, Wifi, WifiOff } from 'lucide-react'

interface Props {
  connected: boolean
  phase: string
  stepCount: number
}

export function Header({ connected, phase, stepCount }: Props) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-luna-border/50">
      {/* Logo */}
      <div className="flex items-center gap-3">
        <motion.div
          className="relative w-8 h-8 flex items-center justify-center"
          animate={{ rotate: 360 }}
          transition={{ duration: 60, repeat: Infinity, ease: 'linear' }}
        >
          <div className="absolute inset-0 rounded-full border border-luna-primary/30" />
          <div className="absolute inset-1 rounded-full border border-luna-primary/50" />
          <span className="text-luna-primary font-mono text-sm font-bold">Λ</span>
        </motion.div>
        <div>
          <h1 className="text-sm font-semibold tracking-wide text-luna-text">
            LUNA
            <span className="text-luna-text-muted font-normal ml-2">Consciousness Engine</span>
          </h1>
          <div className="text-[9px] text-luna-text-muted tracking-widest uppercase">
            v5.1.0 — Dashboard
          </div>
        </div>
      </div>

      {/* Status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-[10px]">
          <Activity className="w-3.5 h-3.5 text-luna-text-dim" />
          <span className="text-luna-text-dim">Step</span>
          <span className="font-mono text-luna-text">{stepCount}</span>
        </div>

        <div className={`phase-badge phase-${phase}`}>
          {phase}
        </div>

        <div className="flex items-center gap-1.5">
          {connected ? (
            <>
              <motion.div
                className="w-2 h-2 rounded-full bg-emerald-400"
                animate={{ opacity: [1, 0.5, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <Wifi className="w-3.5 h-3.5 text-emerald-400" />
            </>
          ) : (
            <>
              <div className="w-2 h-2 rounded-full bg-luna-text-muted" />
              <WifiOff className="w-3.5 h-3.5 text-luna-text-muted" />
              <span className="text-[9px] text-luna-text-muted">Mock</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
