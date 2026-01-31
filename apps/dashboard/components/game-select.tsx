'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { ChevronDown, Gamepad2 } from 'lucide-react'
import { formatBoxArtUrl, type Game } from '@/lib/api'
import { cn } from '@/lib/utils'

interface GameSelectProps {
  games: Game[]
  currentGameId: string | null
}

export function GameSelect({ games, currentGameId }: GameSelectProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)
  
  const selectedGame = games.find(g => g.game_id === currentGameId)
  
  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])
  
  const handleSelect = (gameId: string | null) => {
    const params = new URLSearchParams(searchParams.toString())
    
    if (gameId) {
      params.set('game', gameId)
    } else {
      params.delete('game')
    }
    
    // Remove cursor when changing filters
    params.delete('cursor')
    
    router.push(`/clips?${params.toString()}`)
    setIsOpen(false)
  }
  
  return (
    <div ref={dropdownRef} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 bg-secondary text-sm rounded-lg px-3 py-1.5 hover:bg-secondary/80 transition-colors"
      >
        {selectedGame?.box_art_url ? (
          <img 
            src={formatBoxArtUrl(selectedGame.box_art_url, 20, 27) || ''} 
            alt="" 
            className="w-5 h-5 rounded object-cover"
          />
        ) : (
          <Gamepad2 className="w-4 h-4 text-muted-foreground" />
        )}
        <span>{selectedGame?.game_name || 'All Games'}</span>
        <ChevronDown className={cn('w-4 h-4 transition-transform', isOpen && 'rotate-180')} />
      </button>
      
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-popover border rounded-lg shadow-lg z-50 py-1 max-h-80 overflow-y-auto">
          {/* All Games option */}
          <button
            onClick={() => handleSelect(null)}
            className={cn(
              'w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-muted transition-colors',
              !currentGameId && 'bg-muted'
            )}
          >
            <div className="w-8 h-8 rounded bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center">
              <Gamepad2 className="w-4 h-4 text-white" />
            </div>
            <div className="flex-1 text-left">
              <div className="font-medium">All Games</div>
            </div>
          </button>
          
          {/* Game options */}
          {games.map((game) => {
            const boxArtUrl = formatBoxArtUrl(game.box_art_url, 32, 43)
            
            return (
              <button
                key={game.game_id}
                onClick={() => handleSelect(game.game_id)}
                className={cn(
                  'w-full flex items-center gap-3 px-3 py-2 text-sm hover:bg-muted transition-colors',
                  currentGameId === game.game_id && 'bg-muted'
                )}
              >
                <div className="w-8 h-8 rounded bg-secondary flex items-center justify-center overflow-hidden">
                  {boxArtUrl ? (
                    <img src={boxArtUrl} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <Gamepad2 className="w-4 h-4 text-muted-foreground" />
                  )}
                </div>
                <div className="flex-1 text-left">
                  <div className="font-medium truncate">{game.game_name}</div>
                  <div className="text-xs text-muted-foreground">{game.count} clips</div>
                </div>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
