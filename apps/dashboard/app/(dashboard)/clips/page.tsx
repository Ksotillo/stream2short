import { getSession } from '@/lib/auth'
import { getJobs } from '@/lib/api'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { CreateClipButton } from '@/components/create-clip-modal'
import Link from 'next/link'
import { formatRelativeTime, formatDate } from '@/lib/utils'
import { Film, Clock, CheckCircle, XCircle, RefreshCw, Filter } from 'lucide-react'

export const dynamic = 'force-dynamic'

interface PageProps {
  searchParams: {
    status?: string
    review?: string
    cursor?: string
  }
}

export default async function ClipsPage({ searchParams }: PageProps) {
  const session = await getSession()
  if (!session) return null
  
  const { status, review, cursor } = searchParams
  
  let jobs: Awaited<ReturnType<typeof getJobs>>['jobs'] = []
  let pagination = { next_cursor: null as string | null, has_more: false }
  let error = ''
  
  try {
    const res = await getJobs({
      channel_id: session.id,
      status: status || undefined,
      review_status: review || undefined,
      limit: 20,
      cursor: cursor || undefined,
    })
    jobs = res.jobs
    pagination = res.pagination
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load clips'
  }
  
  return (
    <div className="space-y-6">
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
        <FilterButton href="/clips" active={!status && !review}>
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
              {status || review ? 'Try adjusting your filters' : 'Create your first clip using !clip in chat'}
            </p>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Clips Grid */}
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {jobs.map((job) => (
              <ClipCard key={job.id} job={job} />
            ))}
          </div>
          
          {/* Pagination */}
          {pagination.has_more && (
            <div className="flex justify-center pt-4">
              <Link
                href={`/clips?${new URLSearchParams({
                  ...(status ? { status } : {}),
                  ...(review ? { review } : {}),
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

function ClipCard({ job }: { job: any }) {
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
          <div className="relative aspect-video bg-secondary flex items-center justify-center">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-blue-500/10" />
            <Film className="w-12 h-12 text-muted-foreground/50" />
            
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

