import { getSession } from '@/lib/auth'
import { getJobs, getGames } from '@/lib/api'
import { ClipCard, ClipGrid } from '@/components/clip-card'
import { GameFilters } from '@/components/game-filters'
import { StatusFilters } from '@/components/status-filters'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { Film, Search, Plus, ChevronDown } from 'lucide-react'

export const dynamic = 'force-dynamic'

interface PageProps {
  searchParams: {
    status?: string
    game?: string
    cursor?: string
  }
}

export default async function ClipsPage({ searchParams }: PageProps) {
  const session = await getSession()
  if (!session) return null
  
  const { status, game, cursor } = searchParams
  
  let jobs: Awaited<ReturnType<typeof getJobs>>['jobs'] = []
  let games: Awaited<ReturnType<typeof getGames>>['games'] = []
  let pagination = { next_cursor: null as string | null, has_more: false }
  let error = ''
  
  try {
    // Fetch jobs and games in parallel
    const [jobsRes, gamesRes] = await Promise.all([
      getJobs({
        channel_id: session.id,
        status: status || undefined,
        game_id: game || undefined,
        limit: 20,
        cursor: cursor || undefined,
      }),
      getGames(session.id),
    ])
    
    jobs = jobsRes.jobs
    pagination = jobsRes.pagination
    games = gamesRes.games
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load clips'
  }
  
  return (
    <div className="min-h-screen">
      {/* Header - Mobile optimized */}
      <div className="sticky top-0 z-40 bg-black/80 backdrop-blur-xl border-b border-white/5">
        {/* Search bar */}
        <div className="px-4 pt-4 pb-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-white/40" />
            <input
              type="text"
              placeholder="Search clips..."
              className="w-full pl-10 pr-4 py-2.5 bg-white/5 border border-white/10 rounded-xl text-sm text-white placeholder:text-white/40 focus:outline-none focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/20"
            />
          </div>
        </div>
        
        {/* Game filters - Horizontal scroll */}
        {games.length > 0 && (
          <div className="pb-3">
            <GameFilters 
              games={games} 
              selectedGameId={game || null} 
            />
          </div>
        )}
        
        {/* Status filters */}
        <div className="pb-3 border-b border-white/5">
          <StatusFilters currentStatus={status || null} />
        </div>
      </div>
      
      {/* Content */}
      <div className="py-4">
        {error ? (
          <div className="flex flex-col items-center justify-center py-20 px-4">
            <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center mb-4">
              <Film className="w-8 h-8 text-red-400" />
            </div>
            <p className="text-white/60 text-center">{error}</p>
          </div>
        ) : jobs.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 px-4">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center mb-4">
              <Film className="w-10 h-10 text-violet-400" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-1">No clips yet</h3>
            <p className="text-white/50 text-center text-sm mb-6">
              {status || game 
                ? 'Try changing your filters' 
                : 'Create your first clip using !clip in chat or the button below'}
            </p>
            <Link href="/clips/create">
              <Button className="bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white rounded-full px-6">
                <Plus className="w-4 h-4 mr-2" />
                Create Clip
              </Button>
            </Link>
          </div>
        ) : (
          <>
            {/* Clips Grid */}
            <ClipGrid>
              {jobs.map((job, index) => (
                <ClipCard key={job.id} job={job} index={index} />
              ))}
            </ClipGrid>
            
            {/* Load more */}
            {pagination.has_more && (
              <div className="flex justify-center pt-6 pb-4 px-4">
                <Link
                  href={`/clips?${new URLSearchParams({
                    ...(status ? { status } : {}),
                    ...(game ? { game } : {}),
                    cursor: pagination.next_cursor!,
                  }).toString()}`}
                >
                  <Button 
                    variant="outline" 
                    className="rounded-full border-white/10 text-white/70 hover:text-white hover:bg-white/5"
                  >
                    <ChevronDown className="w-4 h-4 mr-2" />
                    Load more
                  </Button>
                </Link>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
