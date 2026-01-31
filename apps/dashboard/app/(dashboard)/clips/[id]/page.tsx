import { getSession } from '@/lib/auth'
import { getJob } from '@/lib/api'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { formatDate } from '@/lib/utils'
import { ClipActions } from './ClipActions'
import { TranscriptEditor } from '@/components/transcript-editor'
import { 
  ArrowLeft, 
  ExternalLink, 
  Clock, 
  User, 
  Film, 
  AlertCircle,
  CheckCircle,
  XCircle,
  Play,
} from 'lucide-react'

export const dynamic = 'force-dynamic'

interface PageProps {
  params: { id: string }
}

export default async function ClipDetailPage({ params }: PageProps) {
  const session = await getSession()
  if (!session) return null
  
  const { id } = params
  
  let job: Awaited<ReturnType<typeof getJob>>['job'] | null = null
  let events: Awaited<ReturnType<typeof getJob>>['events'] = []
  let error = ''
  
  try {
    const res = await getJob(id, true)
    job = res.job
    events = res.events || []
    
    // Security: Ensure user can only see their own clips
    if (job.channel_id !== session.id) {
      error = 'Clip not found'
      job = null
    }
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load clip'
  }
  
  if (error || !job) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertCircle className="w-12 h-12 text-destructive mb-4" />
        <h2 className="text-xl font-semibold mb-2">Error</h2>
        <p className="text-muted-foreground mb-4">{error || 'Clip not found'}</p>
        <Link href="/clips">
          <Button variant="outline" className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            Back to Clips
          </Button>
        </Link>
      </div>
    )
  }
  
  return (
    <div className="space-y-6">
      {/* Back button */}
      <Link href="/clips" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
        <ArrowLeft className="w-4 h-4" />
        Back to Clips
      </Link>
      
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-2xl font-bold">Clip Details</h1>
            <StatusBadge status={job.status} />
            {job.review_status && (
              <ReviewBadge status={job.review_status} />
            )}
          </div>
          <p className="text-sm text-muted-foreground font-mono">{job.id}</p>
        </div>
        
        <ClipActions job={job} />
      </div>
      
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Video Previews */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Video Preview</h2>
          
          {job.status === 'ready' && (job.final_video_url || job.no_subtitles_url) ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {/* With Subtitles */}
              {job.final_video_url && (
                <VideoPreviewCard
                  title="With Subtitles"
                  url={job.final_video_url}
                />
              )}
              
              {/* Without Subtitles */}
              {job.no_subtitles_url && (
                <VideoPreviewCard
                  title="Without Subtitles"
                  url={job.no_subtitles_url}
                />
              )}
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                {job.status === 'failed' ? (
                  <>
                    <XCircle className="w-12 h-12 mx-auto mb-3 text-destructive" />
                    <p className="text-muted-foreground">Processing failed</p>
                  </>
                ) : (
                  <>
                    <Clock className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
                    <p className="text-muted-foreground">Video still processing...</p>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </div>
        
        {/* Details */}
        <div className="space-y-4">
          {/* Info Card */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Details</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <DetailRow 
                icon={Clock} 
                label="Created" 
                value={formatDate(job.created_at)} 
              />
              <DetailRow 
                icon={User} 
                label="Requested by" 
                value={job.requested_by || 'System'} 
              />
              <DetailRow 
                icon={Film} 
                label="Preset" 
                value={job.render_preset || 'default'} 
              />
              {job.twitch_clip_url && (
                <div className="pt-2">
                  <a
                    href={job.twitch_clip_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-sm text-primary hover:underline"
                  >
                    View original on Twitch
                    <ExternalLink className="w-3 h-3" />
                  </a>
                </div>
              )}
            </CardContent>
          </Card>
          
          {/* Review Notes */}
          {job.review_notes && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Review Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{job.review_notes}</p>
                {job.reviewed_at && (
                  <p className="text-xs text-muted-foreground mt-2">
                    Reviewed on {formatDate(job.reviewed_at)}
                  </p>
                )}
              </CardContent>
            </Card>
          )}
          
          {/* Error */}
          {job.error && (
            <Card className="border-destructive/50">
              <CardHeader className="pb-3">
                <CardTitle className="text-base text-destructive">Error</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm text-destructive whitespace-pre-wrap font-mono bg-destructive/5 p-3 rounded-lg">
                  {job.error}
                </pre>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
      
      {/* Transcript Editor */}
      {(job.transcript_text || job.transcript_segments) && (
        <TranscriptEditor
          jobId={job.id}
          segments={job.transcript_segments}
          transcriptText={job.transcript_text}
          editedAt={job.transcript_edited_at}
          canEdit={job.status === 'ready'}
        />
      )}
      
      {/* Processing Log */}
      {events && events.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Processing Log</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {events.map((event) => (
                <div
                  key={event.id}
                  className={`flex gap-3 text-sm p-2 rounded-lg ${
                    event.level === 'error' ? 'bg-destructive/10' :
                    event.level === 'warn' ? 'bg-yellow-500/10' :
                    'bg-muted/50'
                  }`}
                >
                  <span className="text-xs text-muted-foreground font-mono shrink-0">
                    {new Date(event.created_at).toLocaleTimeString()}
                  </span>
                  {event.stage && (
                    <span className="text-xs font-medium text-muted-foreground shrink-0">
                      [{event.stage}]
                    </span>
                  )}
                  <span className={
                    event.level === 'error' ? 'text-destructive' :
                    event.level === 'warn' ? 'text-yellow-500' :
                    'text-foreground'
                  }>
                    {event.message}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  const config = {
    ready: { icon: CheckCircle, variant: 'success' as const, label: 'Ready' },
    failed: { icon: XCircle, variant: 'destructive' as const, label: 'Failed' },
    queued: { icon: Clock, variant: 'warning' as const, label: 'Queued' },
  }[status] || { icon: Clock, variant: 'info' as const, label: status }
  
  return (
    <Badge variant={config.variant} className="gap-1">
      <config.icon className="w-3 h-3" />
      {config.label}
    </Badge>
  )
}

function ReviewBadge({ status }: { status: string }) {
  if (status === 'pending') return null
  
  return (
    <Badge variant={status === 'approved' ? 'success' : 'destructive'}>
      {status}
    </Badge>
  )
}

function DetailRow({ 
  icon: Icon, 
  label, 
  value 
}: { 
  icon: React.ElementType
  label: string
  value: string 
}) {
  return (
    <div className="flex items-center gap-3">
      <Icon className="w-4 h-4 text-muted-foreground" />
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className="text-sm font-medium ml-auto">{value}</span>
    </div>
  )
}

function VideoPreviewCard({ title, url }: { title: string; url: string }) {
  const isGoogleDrive = url.includes('drive.google.com')
  
  return (
    <Card className="overflow-hidden">
      <CardContent className="p-0">
        <div className="aspect-[9/16] bg-black flex items-center justify-center relative group">
          {isGoogleDrive ? (
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-primary/20 to-primary/5 group-hover:from-primary/30 group-hover:to-primary/10 transition-colors"
            >
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center group-hover:scale-110 transition-transform">
                <Play className="w-8 h-8 text-primary ml-1" />
              </div>
            </a>
          ) : (
            <video
              src={url}
              controls
              className="w-full h-full object-contain"
            />
          )}
        </div>
        <div className="p-3 border-t">
          <p className="text-sm font-medium">{title}</p>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary hover:underline inline-flex items-center gap-1 mt-1"
          >
            Open in new tab
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>
      </CardContent>
    </Card>
  )
}

