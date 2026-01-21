import { getSession } from '@/lib/auth'
import { getJobs, getGames } from '@/lib/api'
import { ClipCard, ClipGrid } from '@/components/clip-card'
import { GameFilters } from '@/components/game-filters'
import { StatusFilters } from '@/components/status-filters'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { CreateClipButton } from '@/components/create-clip-modal'
import Link from 'next/link'
import { formatRelativeTime, formatDate } from '@/lib/utils'
import { Film, Search, Plus, ChevronDown, CheckCircle, XCircle, RefreshCw, Clock, Filter } from 'lucide-react'

export const dynamic = 'force-dynamic'

interface PageProps {
  searchParams: {
    status?: string
    game?: string
    review?: string
    cursor?: string
  }
}

export default async function ClipsPage({ searchParams }: PageProps) {
  const session = await getSession()
  if (!session) return null
  
  const { status, game, review, cursor } = searchParams
  
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
        review_status: review || undefined,
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
    <>
      {/* ==================== MOBILE VERSION ==================== */}
      <div className="lg:hidden min-h-screen">
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
      
      {/* ==================== DESKTOP VERSION ==================== */}
      <div className="hidden lg:block space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Your Clips</h1>
            <p className="text-muted-foreground">
              Manage and review all your processed clips
            </p>
          </div>
          
          <CreateClipButton />
        </div>
        
        {/* Filters */}
        <div className="flex flex-wrap gap-2">
          <FilterButton href="/clips" active={!status && !review && !game}>
            All
          </FilterButton>
          <FilterButton href="/clips?status=ready" active={status === 'ready'}>
            <CheckCircle className="w-3 h-3" />
            Ready
          </FilterButton>
          <FilterButton href="/clips?status=failed" active={status === 'failed'}>
            <XCircle className="w-3 h-3" />
            Failed
          </FilterButton>
          <FilterButton 
            href="/clips?status=queued,downloading,transcribing,rendering,uploading" 
            active={status?.includes('queued') ?? false}
          >
            <RefreshCw className="w-3 h-3" />
            Processing
          </FilterButton>
          <FilterButton href="/clips?review=pending" active={review === 'pending'}>
            Needs Review
          </FilterButton>
          
          {/* Game filters dropdown for desktop */}
          {games.length > 0 && (
            <div className="flex items-center gap-2 ml-auto">
              <Filter className="w-4 h-4 text-muted-foreground" />
              <select 
                className="bg-secondary text-sm rounded-lg px-3 py-1.5 border-0 focus:ring-2 focus:ring-primary"
                value={game || ''}
                onChange={(e) => {
                  const newGame = e.target.value
                  const params = new URLSearchParams()
                  if (status) params.set('status', status)
                  if (review) params.set('review', review)
                  if (newGame) params.set('game', newGame)
                  window.location.href = `/clips?${params.toString()}`
                }}
              >
                <option value="">All Games</option>
                {games.map((g) => (
                  <option key={g.game_id} value={g.game_id}>
                    {g.game_name} ({g.count})
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        
        {error ? (
          <Card className="border-destructive/50">
            <CardContent className="p-6 text-center text-destructive">
              {error}
            </CardContent>
          </Card>
        ) : jobs.length === 0 ? (
          <Card>
            <CardContent className="p-12 text-center">
              <Film className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-1">No clips found</h3>
              <p className="text-muted-foreground text-sm">
                {status || review || game ? 'Try adjusting your filters' : 'Create your first clip using !clip in chat'}
              </p>
            </CardContent>
          </Card>
        ) : (
          <>
            {/* Desktop Clips Grid */}
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {jobs.map((job) => (
                <DesktopClipCard key={job.id} job={job} />
              ))}
            </div>
            
            {/* Pagination */}
            {pagination.has_more && (
              <div className="flex justify-center pt-4">
                <Link
                  href={`/clips?${new URLSearchParams({
                    ...(status ? { status } : {}),
                    ...(review ? { review } : {}),
                    ...(game ? { game } : {}),
                    cursor: pagination.next_cursor!,
                  }).toString()}`}
                >
                  <Button variant="outline">Load more</Button>
                </Link>
              </div>
            )}
          </>
        )}
      </div>
    </>
  )
}

function FilterButton({
  href,
  active,
  children,
}: {
  href: string
  active: boolean
  children: React.ReactNode
}) {
  return (
    <Link href={href}>
      <Button
        variant={active ? 'secondary' : 'ghost'}
        size="sm"
        className="gap-1.5"
      >
        {children}
      </Button>
    </Link>
  )
}

function DesktopClipCard({ job }: { job: any }) {
  const getStatusIcon = () => {
    switch (job.status) {
      case 'ready':
        return <CheckCircle className="w-4 h-4 text-emerald-400" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-400" />
      default:
        return <RefreshCw className="w-4 h-4 text-blue-400 animate-spin" />
    }
  }
  
  return (
    <Link href={`/clips/${job.id}`}>
      <Card className="group hover:border-primary/50 transition-all duration-200 cursor-pointer overflow-hidden">
        <CardContent className="p-0">
          {/* Video preview area */}
          <div className="relative aspect-video bg-secondary flex items-center justify-center overflow-hidden">
            {job.thumbnail_url ? (
              <img 
                src={job.thumbnail_url} 
                alt="" 
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
            ) : (
              <>
                <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-blue-500/10" />
                <Film className="w-12 h-12 text-muted-foreground/50" />
              </>
            )}
            
            {/* Status badge overlay */}
            <div className="absolute top-2 right-2">
              <Badge 
                variant={
                  job.status === 'ready' ? 'success' : 
                  job.status === 'failed' ? 'destructive' : 
                  'info'
                }
                className="gap-1"
              >
                {getStatusIcon()}
                {job.status}
              </Badge>
            </div>
            
            {/* Game badge */}
            {job.game_name && (
              <div className="absolute bottom-2 left-2">
                <Badge variant="secondary" className="bg-black/60 text-white text-xs">
                  {job.game_name}
                </Badge>
              </div>
            )}
          </div>
          
          {/* Info */}
          <div className="p-4 space-y-3">
            <div className="flex items-start justify-between gap-2">
              <p className="text-sm font-medium line-clamp-2 flex-1">
                {job.transcript_text?.slice(0, 60) || 'Processing clip...'}
                {job.transcript_text?.length > 60 && '...'}
              </p>
            </div>
            
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>{formatDate(job.created_at)}</span>
              {job.review_status && job.review_status !== 'pending' && (
                <Badge 
                  variant={job.review_status === 'approved' ? 'success' : 'destructive'}
                  className="text-[10px] h-5"
                >
                  {job.review_status}
                </Badge>
              )}
            </div>
            
            {job.error && (
              <p className="text-xs text-destructive line-clamp-1">
                {job.error}
              </p>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}
