import { getSession } from '@/lib/auth'
import { getJobs } from '@/lib/api'
import { ClipCard } from '@/components/clip-card'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { CreateClipButton } from '@/components/create-clip-modal'
import Link from 'next/link'
import { formatRelativeTime, formatDate } from '@/lib/utils'
import { 
  Film, 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  ArrowRight,
  Sparkles,
  TrendingUp,
  Zap,
  RefreshCw,
  XCircle,
} from 'lucide-react'

export const dynamic = 'force-dynamic'

export default async function DashboardPage() {
  const session = await getSession()
  
  if (!session) return null
  
  let recentJobs: Awaited<ReturnType<typeof getJobs>>['jobs'] = []
  let error = ''
  
  try {
    const res = await getJobs({ channel_id: session.id, limit: 10 })
    recentJobs = res.jobs
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load jobs'
  }
  
  // Calculate stats
  const stats = {
    total: recentJobs.length,
    ready: recentJobs.filter(j => j.status === 'ready').length,
    processing: recentJobs.filter(j => !['ready', 'failed', 'queued'].includes(j.status)).length,
    failed: recentJobs.filter(j => j.status === 'failed').length,
    pending_review: recentJobs.filter(j => j.status === 'ready' && j.review_status === 'pending').length,
  }
  
  return (
    <>
      {/* ==================== MOBILE VERSION ==================== */}
      <div className="lg:hidden min-h-screen">
        {/* Welcome Section */}
        <div className="px-4 pt-6 pb-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="relative">
              {session.profile_image_url ? (
                <img 
                  src={session.profile_image_url} 
                  alt={session.display_name}
                  className="w-14 h-14 rounded-2xl object-cover ring-2 ring-violet-500/50"
                />
              ) : (
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-xl font-bold text-white">
                  {session.display_name[0]}
                </div>
              )}
              <div className="absolute -bottom-1 -right-1 w-5 h-5 bg-emerald-500 rounded-full border-2 border-black" />
            </div>
            <div>
              <p className="text-white/60 text-sm">Welcome back,</p>
              <h1 className="text-xl font-bold text-white">{session.display_name}</h1>
            </div>
          </div>
          
          {/* Quick action card */}
          <Link href="/clips/create">
            <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-violet-600 to-fuchsia-600 p-6">
              <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2" />
              <div className="relative z-10">
                <div className="flex items-center gap-2 mb-3">
                  <Sparkles className="w-5 h-5 text-white/90" />
                  <span className="text-sm font-medium text-white/90">Quick Action</span>
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">Create a new Short</h2>
                <p className="text-white/70 text-sm mb-4">Transform your Twitch clips into viral shorts</p>
                <div className="inline-flex items-center gap-2 text-white font-medium">
                  Get started
                  <ArrowRight className="w-4 h-4" />
                </div>
              </div>
            </div>
          </Link>
        </div>
        
        {/* Stats Section */}
        <div className="px-4 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">Your Stats</h2>
          <div className="grid grid-cols-2 gap-3">
            <MobileStatCard
              icon={Film}
              label="Total Clips"
              value={stats.total}
              gradient="from-blue-500/20 to-cyan-500/20"
              iconColor="text-blue-400"
            />
            <MobileStatCard
              icon={CheckCircle}
              label="Ready"
              value={stats.ready}
              gradient="from-emerald-500/20 to-green-500/20"
              iconColor="text-emerald-400"
            />
            <MobileStatCard
              icon={Clock}
              label="Processing"
              value={stats.processing}
              gradient="from-amber-500/20 to-yellow-500/20"
              iconColor="text-amber-400"
            />
            <MobileStatCard
              icon={AlertCircle}
              label="Needs Review"
              value={stats.pending_review}
              gradient="from-violet-500/20 to-purple-500/20"
              iconColor="text-violet-400"
            />
          </div>
        </div>
        
        {/* Recent Clips */}
        <div className="px-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Recent Clips</h2>
            <Link href="/clips">
              <Button 
                variant="ghost" 
                size="sm"
                className="text-white/60 hover:text-white hover:bg-white/5 -mr-2"
              >
                See all
                <ArrowRight className="w-4 h-4 ml-1" />
              </Button>
            </Link>
          </div>
          
          {error ? (
            <div className="rounded-2xl bg-red-500/10 border border-red-500/20 p-6 text-center">
              <AlertCircle className="w-8 h-8 mx-auto mb-2 text-red-400" />
              <p className="text-red-400">{error}</p>
            </div>
          ) : recentJobs.length === 0 ? (
            <div className="rounded-2xl bg-white/5 border border-white/10 p-8 text-center">
              <div className="w-16 h-16 rounded-full bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center mx-auto mb-4">
                <Film className="w-8 h-8 text-violet-400" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-1">No clips yet</h3>
              <p className="text-white/50 text-sm mb-4">
                Use !clip in chat or create one manually
              </p>
              <Link href="/clips/create">
                <Button className="bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white rounded-full">
                  Create your first clip
                </Button>
              </Link>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3">
              {recentJobs.slice(0, 4).map((job, index) => (
                <ClipCard key={job.id} job={job} index={index} />
              ))}
            </div>
          )}
        </div>
      </div>
      
      {/* ==================== DESKTOP VERSION ==================== */}
      <div className="hidden lg:block space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Welcome back, {session.display_name}
          </h1>
          <p className="text-muted-foreground mt-1">
            Here's what's happening with your clips
          </p>
        </div>
        
        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <DesktopStatsCard
            title="Total Clips"
            value={stats.total}
            icon={Film}
            description="All time"
          />
          <DesktopStatsCard
            title="Ready"
            value={stats.ready}
            icon={CheckCircle}
            description="Completed clips"
            className="text-emerald-400"
          />
          <DesktopStatsCard
            title="Processing"
            value={stats.processing}
            icon={Clock}
            description="Currently working"
            className="text-blue-400"
          />
          <DesktopStatsCard
            title="Needs Review"
            value={stats.pending_review}
            icon={AlertCircle}
            description="Pending approval"
            className="text-yellow-400"
          />
        </div>
        
        {/* Recent Clips */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Recent Clips</h2>
            <div className="flex items-center gap-2">
              <CreateClipButton />
              <Link href="/clips">
                <Button variant="ghost" className="gap-1">
                  View all
                  <ArrowRight className="w-4 h-4" />
                </Button>
              </Link>
            </div>
          </div>
          
          {error ? (
            <Card className="border-destructive/50">
              <CardContent className="p-6 text-center text-destructive">
                <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                <p>{error}</p>
              </CardContent>
            </Card>
          ) : recentJobs.length === 0 ? (
            <Card>
              <CardContent className="p-12 text-center">
                <Film className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                <h3 className="text-lg font-medium mb-1">No clips yet</h3>
                <p className="text-muted-foreground text-sm">
                  Use the !clip command in your Twitch chat to create your first clip
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {recentJobs.slice(0, 6).map((job) => (
                <DesktopClipCard key={job.id} job={job} />
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  )
}

// Mobile components
function MobileStatCard({
  icon: Icon,
  label,
  value,
  gradient,
  iconColor,
}: {
  icon: React.ElementType
  label: string
  value: number
  gradient: string
  iconColor: string
}) {
  return (
    <div className={`rounded-2xl bg-gradient-to-br ${gradient} border border-white/5 p-4`}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${iconColor}`} />
        <span className="text-xs text-white/60 font-medium">{label}</span>
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  )
}

// Desktop components
function DesktopStatsCard({
  title,
  value,
  icon: Icon,
  description,
  className,
}: {
  title: string
  value: number
  icon: React.ElementType
  description: string
  className?: string
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        <Icon className={`w-5 h-5 ${className || 'text-muted-foreground'}`} />
      </CardHeader>
      <CardContent>
        <div className="text-3xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </CardContent>
    </Card>
  )
}

function DesktopClipCard({ job }: { job: any }) {
  const statusVariantMap: Record<string, string> = {
    ready: 'success',
    failed: 'destructive',
    queued: 'warning',
  }
  const statusVariant = statusVariantMap[job.status as string] || 'info'
  
  const getStatusIcon = () => {
    switch (job.status) {
      case 'ready':
        return <CheckCircle className="w-3 h-3" />
      case 'failed':
        return <XCircle className="w-3 h-3" />
      default:
        return <RefreshCw className="w-3 h-3 animate-spin" />
    }
  }
  
  return (
    <Link href={`/clips/${job.id}`}>
      <Card className="group hover:border-primary/50 transition-colors cursor-pointer overflow-hidden">
        <CardContent className="p-0">
          {/* Thumbnail */}
          <div className="aspect-video bg-secondary flex items-center justify-center overflow-hidden relative">
            {job.thumbnail_url ? (
              <img 
                src={job.thumbnail_url} 
                alt="" 
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
            ) : job.final_video_url ? (
              <div className="w-full h-full bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center">
                <Film className="w-8 h-8 text-primary/60" />
              </div>
            ) : (
              <Clock className="w-8 h-8 text-muted-foreground animate-pulse" />
            )}
            
            {/* Game badge */}
            {job.game_name && (
              <div className="absolute bottom-2 left-2">
                <Badge variant="secondary" className="bg-black/60 text-white text-xs">
                  {job.game_name}
                </Badge>
              </div>
            )}
          </div>
          
          <div className="p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant={statusVariant as any} className="gap-1">
                {getStatusIcon()}
                {job.status}
              </Badge>
              {job.review_status && job.review_status !== 'pending' && (
                <Badge variant={job.review_status === 'approved' ? 'success' : 'destructive'}>
                  {job.review_status}
                </Badge>
              )}
            </div>
            
            <p className="text-sm text-muted-foreground line-clamp-2">
              {job.transcript_text?.slice(0, 80) || 'Processing...'}
              {job.transcript_text?.length > 80 && '...'}
            </p>
            
            <p className="text-xs text-muted-foreground">
              {formatRelativeTime(job.created_at)}
            </p>
          </div>
        </CardContent>
      </Card>
    </Link>
  )
}
