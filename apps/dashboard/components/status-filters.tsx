'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface StatusFiltersProps {
  currentStatus: string | null
}

const filters = [
  { value: null, label: 'All' },
  { value: 'ready', label: 'Ready' },
  { value: 'queued,downloading,transcribing,rendering,uploading', label: 'Processing' },
  { value: 'failed', label: 'Failed' },
]

export function StatusFilters({ currentStatus }: StatusFiltersProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const handleStatusChange = (status: string | null) => {
    const params = new URLSearchParams(searchParams.toString())
    
    if (status === null) {
      params.delete('status')
    } else {
      params.set('status', status)
    }
    
    // Reset cursor when changing filters
    params.delete('cursor')
    
    router.push(`/clips?${params.toString()}`)
  }
  
  return (
    <div className="flex gap-2 px-4 overflow-x-auto scrollbar-hide -mx-4 px-4">
      {filters.map((filter) => {
        const isActive = currentStatus === filter.value
        
        return (
          <motion.button
            key={filter.label}
            whileTap={{ scale: 0.95 }}
            onClick={() => handleStatusChange(filter.value)}
            className={cn(
              'relative px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-colors',
              isActive
                ? 'text-white'
                : 'text-white/60 hover:text-white/80'
            )}
          >
            {isActive && (
              <motion.div
                layoutId="statusPill"
                className="absolute inset-0 bg-gradient-to-r from-violet-500/30 to-fuchsia-500/30 border border-violet-500/50 rounded-full"
                transition={{ type: 'spring', stiffness: 500, damping: 30 }}
              />
            )}
            <span className="relative z-10">{filter.label}</span>
          </motion.button>
        )
      })}
    </div>
  )
}

