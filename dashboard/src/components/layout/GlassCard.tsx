// Luna Dashboard — CC-BY-NC-4.0 — (c) Varden
// https://creativecommons.org/licenses/by-nc/4.0/
import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

interface Props {
  title: string
  subtitle?: string
  icon?: ReactNode
  children: ReactNode
  className?: string
  glow?: boolean
}

export function GlassCard({ title, subtitle, icon, children, className = '', glow }: Props) {
  return (
    <motion.div
      className={`${glow ? 'glass-panel-glow' : 'glass-panel'} p-5 flex flex-col gap-4 ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center gap-2">
        {icon && <span className="text-luna-primary">{icon}</span>}
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-widest text-luna-text-dim">
            {title}
          </h2>
          {subtitle && (
            <span className="text-[9px] text-luna-text-muted">{subtitle}</span>
          )}
        </div>
      </div>
      {children}
    </motion.div>
  )
}
