'use client'

import Link from 'next/link'
import Image from 'next/image'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { formatRelativeTime } from '@/lib/utils'
import { 
  Play, 
  CheckCircle, 
  XCircle, 
  Loader2,
  Gamepad2,
} from 'lucide-react'
import type { Job } from '@/lib/api'

interface ClipCardProps {
  job: Job
  index: number
}

export function ClipCard({ job, index }: ClipCardProps) {
  const isReady = job.status === 'ready'
  const isFailed = job.status === 'failed'
  const isProcessing = !isReady && !isFailed
  
  // Get a preview text (first few words of transcript or status message)
  const previewText = job.transcript_text 
    ? job.transcript_text.slice(0, 50) + (job.transcript_text.length > 50 ? '...' : '')
    : isProcessing 
      ? 'Processing...' 
      : isFailed 
        ? 'Failed to process' 
        : 'No transcript'
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.3, 
        delay: index * 0.05,
        ease: [0.25, 0.1, 0.25, 1]
      }}
    >
      <Link href={`/clips/${job.id}`}>
        <motion.div
          whileTap={{ scale: 0.98 }}
          className="group relative overflow-hidden rounded-2xl bg-gradient-to-b from-white/10 to-white/5 backdrop-blur-sm"
        >
          {/* Thumbnail / Preview area */}
          <div className="relative aspect-[4/5] overflow-hidden">
            {job.thumbnail_url ? (
              <Image
                src={job.thumbnail_url}
                alt={previewText}
                fill
                className="object-cover transition-transform duration-500 group-hover:scale-105"
                sizes="(max-width: 768px) 50vw, 33vw"
              />
            ) : (
              <div className="absolute inset-0 bg-gradient-to-br from-violet-900/50 to-fuchsia-900/50 flex items-center justify-center">
                <Gamepad2 className="w-12 h-12 text-white/30" />
              </div>
            )}
            
            {/* Gradient overlay */}
            <div className="absolute inset-0 bg-gradient-to-t from-black via-black/20 to-transparent opacity-90" />
            
            {/* Status indicator */}
            <div className="absolute top-3 right-3">
              {isProcessing && (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                  className="w-8 h-8 rounded-full bg-blue-500/90 backdrop-blur-sm flex items-center justify-center"
                >
                  <Loader2 className="w-4 h-4 text-white" />
                </motion.div>
              )}
              {isReady && (
                <div className="w-8 h-8 rounded-full bg-emerald-500/90 backdrop-blur-sm flex items-center justify-center">
                  <CheckCircle className="w-4 h-4 text-white" />
                </div>
              )}
              {isFailed && (
                <div className="w-8 h-8 rounded-full bg-red-500/90 backdrop-blur-sm flex items-center justify-center">
                  <XCircle className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
            
            
            {/* Play button overlay */}
            {isReady && (
              <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  whileHover={{ scale: 1, opacity: 1 }}
                  className="w-14 h-14 rounded-full bg-white/20 backdrop-blur-md flex items-center justify-center"
                >
                  <Play className="w-6 h-6 text-white ml-1" fill="white" />
                </motion.div>
              </div>
            )}
            
            {/* Bottom content */}
            <div className="absolute bottom-0 left-0 right-0 p-3">
              {/* Game badge */}
              {job.game_name && (
                <div className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/10 backdrop-blur-sm mb-2">
                  <Gamepad2 className="w-3 h-3 text-violet-400" />
                  <span className="text-[10px] text-white/80 font-medium">
                    {job.game_name}
                  </span>
                </div>
              )}
              
              {/* Title / Transcript preview */}
              <h3 className="text-sm font-semibold text-white line-clamp-2 leading-tight">
                {previewText}
              </h3>
              
              {/* Time */}
              <p className="text-xs text-white/50 mt-1">
                {formatRelativeTime(job.created_at)}
              </p>
            </div>
          </div>
        </motion.div>
      </Link>
    </motion.div>
  )
}

// Grid container with stagger animation
export function ClipGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:gap-4 px-4">
      {children}
    </div>
  )
}

