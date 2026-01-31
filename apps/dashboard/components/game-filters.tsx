'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { Gamepad2, Sparkles } from 'lucide-react'
import { formatBoxArtUrl, type Game } from '@/lib/api'

// Game category icons/colors mapping
const gameStyles: Record<string, { color: string; gradient: string }> = {
  // Popular games - add more as needed
  'fortnite': { color: '#9D4DFF', gradient: 'from-violet-500 to-blue-500' },
  'league of legends': { color: '#C89B3C', gradient: 'from-amber-500 to-yellow-400' },
  'valorant': { color: '#FF4654', gradient: 'from-red-500 to-pink-500' },
  'minecraft': { color: '#52A535', gradient: 'from-green-500 to-emerald-400' },
  'just chatting': { color: '#9147FF', gradient: 'from-purple-500 to-violet-400' },
  'call of duty': { color: '#1B1B1B', gradient: 'from-gray-600 to-gray-800' },
  'apex legends': { color: '#DA292A', gradient: 'from-red-600 to-orange-500' },
  'gta v': { color: '#4CAF50', gradient: 'from-green-500 to-teal-500' },
  'rocket league': { color: '#0078F2', gradient: 'from-blue-500 to-cyan-400' },
  'overwatch 2': { color: '#FA9C1E', gradient: 'from-orange-500 to-amber-400' },
  'counter-strike 2': { color: '#DE752B', gradient: 'from-orange-600 to-yellow-500' },
  'dota 2': { color: '#B8383B', gradient: 'from-red-700 to-red-500' },
  'world of warcraft': { color: '#5A9FD4', gradient: 'from-blue-400 to-blue-600' },
  default: { color: '#6366F1', gradient: 'from-indigo-500 to-purple-500' },
}

function getGameStyle(gameName: string) {
  const key = gameName.toLowerCase()
  return gameStyles[key] || gameStyles.default
}

interface GameFiltersProps {
  games: Game[]
  selectedGameId: string | null
}

export function GameFilters({ games, selectedGameId }: GameFiltersProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const handleGameSelect = (gameId: string | null) => {
    const params = new URLSearchParams(searchParams.toString())
    
    if (gameId === null) {
      params.delete('game')
    } else {
      params.set('game', gameId)
    }
    
    // Reset cursor when changing filters
    params.delete('cursor')
    
    router.push(`/clips?${params.toString()}`)
  }
  
  return (
    <div className="relative">
      {/* Fade edges */}
      <div className="absolute left-0 top-0 bottom-0 w-6 bg-gradient-to-r from-black to-transparent z-10 pointer-events-none" />
      <div className="absolute right-0 top-0 bottom-0 w-6 bg-gradient-to-l from-black to-transparent z-10 pointer-events-none" />
      
      {/* Scrollable container */}
      <div className="flex gap-3 overflow-x-auto scrollbar-hide px-4 py-2 -mx-4 snap-x snap-mandatory">
        {/* All games option */}
        <motion.button
          whileTap={{ scale: 0.95 }}
          onClick={() => handleGameSelect(null)}
          className={cn(
            'flex flex-col items-center gap-1.5 min-w-[72px] snap-start',
          )}
        >
          <div
            className={cn(
              'w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-200',
              selectedGameId === null
                ? 'bg-gradient-to-br from-violet-500 to-fuchsia-500 ring-2 ring-violet-400 ring-offset-2 ring-offset-black'
                : 'bg-white/10 hover:bg-white/15'
            )}
          >
            <Sparkles className={cn(
              'w-6 h-6',
              selectedGameId === null ? 'text-white' : 'text-white/70'
            )} />
          </div>
          <span className={cn(
            'text-xs font-medium truncate max-w-[72px]',
            selectedGameId === null ? 'text-white' : 'text-white/60'
          )}>
            All
          </span>
        </motion.button>
        
        {/* Game filters */}
        {games.map((game) => {
          const isSelected = selectedGameId === game.game_id
          const style = getGameStyle(game.game_name)
          const boxArtUrl = formatBoxArtUrl(game.box_art_url, 56, 75)
          
          return (
            <motion.button
              key={game.game_id}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleGameSelect(game.game_id)}
              className="flex flex-col items-center gap-1.5 min-w-[72px] snap-start"
            >
              <div
                className={cn(
                  'w-14 h-14 rounded-2xl flex items-center justify-center transition-all duration-200 overflow-hidden',
                  isSelected
                    ? 'ring-2 ring-offset-2 ring-offset-black ring-violet-400'
                    : 'ring-1 ring-white/10 hover:ring-white/20',
                  !boxArtUrl && (isSelected
                    ? `bg-gradient-to-br ${style.gradient}`
                    : 'bg-white/10 hover:bg-white/15')
                )}
              >
                {boxArtUrl ? (
                  <img
                    src={boxArtUrl}
                    alt={game.game_name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <Gamepad2 className={cn(
                    'w-6 h-6',
                    isSelected ? 'text-white' : 'text-white/70'
                  )} />
                )}
              </div>
              <span className={cn(
                'text-xs font-medium truncate max-w-[72px] text-center',
                isSelected ? 'text-white' : 'text-white/60'
              )}>
                {game.game_name.length > 10 ? `${game.game_name.slice(0, 9)}...` : game.game_name}
              </span>
              {/* Clip count badge */}
              <span className="text-[10px] text-white/40">
                {game.count} clips
              </span>
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}

