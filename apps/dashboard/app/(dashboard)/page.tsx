import { getSession } from '@/lib/auth'
import { getJobs } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { formatRelativeTime } from '@/lib/utils'
import { Film, CheckCircle, Clock, AlertCircle, ArrowRight } from 'lucide-react'

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
    <div className="space-y-8">
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
        <StatsCard
          title="Total Clips"
          value={stats.total}
          icon={Film}
          description="All time"
        />
        <StatsCard
          title="Ready"
          value={stats.ready}
          icon={CheckCircle}
          description="Completed clips"
          className="text-emerald-400"
        />
        <StatsCard
          title="Processing"
          value={stats.processing}
          icon={Clock}
          description="Currently working"
          className="text-blue-400"
        />
        <StatsCard
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
          <Link href="/clips">
            <Button variant="ghost" className="gap-1">
              View all
              <ArrowRight className="w-4 h-4" />
            </Button>
          </Link>
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
              <ClipCard key={job.id} job={job} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function StatsCard({
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

function ClipCard({ job }: { job: any }) {
  const statusVariantMap: Record<string, string> = {
    ready: 'success',
    failed: 'destructive',
    queued: 'warning',
  }
  const statusVariant = statusVariantMap[job.status as string] || 'info'
  
  return (
    <Link href={`/clips/${job.id}`}>
      <Card className="group hover:border-primary/50 transition-colors cursor-pointer">
        <CardContent className="p-4">
          {/* Thumbnail placeholder */}
          <div className="aspect-[9/16] max-h-32 bg-secondary rounded-lg mb-3 flex items-center justify-center overflow-hidden">
            {job.final_video_url ? (
              <div className="w-full h-full bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center">
                <Film className="w-8 h-8 text-primary/60" />
              </div>
            ) : (
              <Clock className="w-8 h-8 text-muted-foreground animate-pulse" />
            )}
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant={statusVariant as any}>{job.status}</Badge>
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

