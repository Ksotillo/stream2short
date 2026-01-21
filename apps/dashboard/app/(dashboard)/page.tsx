import { getSession } from '@/lib/auth'
import { getJobs } from '@/lib/api'
import { ClipCard } from '@/components/clip-card'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { formatRelativeTime } from '@/lib/utils'
import { 
  Film, 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  ArrowRight,
  Sparkles,
  TrendingUp,
  Zap,
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
    <div className="min-h-screen">
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
          <StatCard
            icon={Film}
            label="Total Clips"
            value={stats.total}
            gradient="from-blue-500/20 to-cyan-500/20"
            iconColor="text-blue-400"
          />
          <StatCard
            icon={CheckCircle}
            label="Ready"
            value={stats.ready}
            gradient="from-emerald-500/20 to-green-500/20"
            iconColor="text-emerald-400"
          />
          <StatCard
            icon={Clock}
            label="Processing"
            value={stats.processing}
            gradient="from-amber-500/20 to-yellow-500/20"
            iconColor="text-amber-400"
          />
          <StatCard
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
  )
}

function StatCard({
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
