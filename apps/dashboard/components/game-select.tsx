'use client'

import { useRouter, useSearchParams } from 'next/navigation'

interface Game {
  game_id: string
  game_name: string
  count: number
}

interface GameSelectProps {
  games: Game[]
  currentGameId: string | null
}

export function GameSelect({ games, currentGameId }: GameSelectProps) {
  const router = useRouter()
  const searchParams = useSearchParams()
  
  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newGame = e.target.value
    const params = new URLSearchParams(searchParams.toString())
    
    if (newGame) {
      params.set('game', newGame)
    } else {
      params.delete('game')
    }
    
    // Remove cursor when changing filters
    params.delete('cursor')
    
    router.push(`/clips?${params.toString()}`)
  }
  
  return (
    <select 
      className="bg-secondary text-sm rounded-lg px-3 py-1.5 border-0 focus:ring-2 focus:ring-primary"
      value={currentGameId || ''}
      onChange={handleChange}
    >
      <option value="">All Games</option>
      {games.map((g) => (
        <option key={g.game_id} value={g.game_id}>
          {g.game_name} ({g.count})
        </option>
      ))}
    </select>
  )
}
