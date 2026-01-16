import { getJob } from '@/lib/api'
import Link from 'next/link'
import { JobActions } from './JobActions'

export const dynamic = 'force-dynamic'

interface PageProps {
  params: { id: string }
}

export default async function JobDetailPage({ params }: PageProps) {
  const { id } = params

  let job: Awaited<ReturnType<typeof getJob>>['job'] | null = null
  let events: Awaited<ReturnType<typeof getJob>>['events'] = []
  let error = ''

  try {
    const res = await getJob(id, true)
    job = res.job
    events = res.events || []
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load job'
  }

  if (error || !job) {
    return (
      <div className="text-center py-12">
        <div className="text-red-400 mb-4">⚠️ {error || 'Job not found'}</div>
        <Link href="/jobs" className="text-twitch-purple hover:underline">
          ← Back to jobs
        </Link>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <Link href="/jobs" className="text-zinc-400 hover:text-white text-sm mb-2 block">
            ← Back to jobs
          </Link>
          <h1 className="text-2xl font-bold flex items-center gap-3">
            Job Details
            <span className={`status-badge status-${job.status}`}>
              {job.status}
            </span>
          </h1>
          <p className="text-zinc-500 font-mono text-sm mt-1">{job.id}</p>
        </div>
        
        <JobActions job={job} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Video Previews */}
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Video Preview</h2>
          
          {job.status === 'ready' && job.final_video_url ? (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* With Subtitles */}
              <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                <h3 className="text-sm font-medium text-zinc-400 mb-2">With Subtitles</h3>
                {job.final_video_url.includes('drive.google.com') ? (
                  <a
                    href={job.final_video_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block aspect-[9/16] bg-zinc-800 rounded-lg flex items-center justify-center hover:bg-zinc-700 transition"
                  >
                    <span className="text-4xl">🎬</span>
                  </a>
                ) : (
                  <video
                    src={job.final_video_url}
                    controls
                    className="w-full aspect-[9/16] bg-black rounded-lg"
                  />
                )}
                <a
                  href={job.final_video_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-twitch-purple hover:underline mt-2 block"
                >
                  Open in Drive →
                </a>
              </div>
              
              {/* Without Subtitles */}
              {job.no_subtitles_url && (
                <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-zinc-400 mb-2">Without Subtitles</h3>
                  {job.no_subtitles_url.includes('drive.google.com') ? (
                    <a
                      href={job.no_subtitles_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block aspect-[9/16] bg-zinc-800 rounded-lg flex items-center justify-center hover:bg-zinc-700 transition"
                    >
                      <span className="text-4xl">🎬</span>
                    </a>
                  ) : (
                    <video
                      src={job.no_subtitles_url}
                      controls
                      className="w-full aspect-[9/16] bg-black rounded-lg"
                    />
                  )}
                  <a
                    href={job.no_subtitles_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-twitch-purple hover:underline mt-2 block"
                  >
                    Open in Drive →
                  </a>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">
              <p className="text-zinc-400">
                {job.status === 'failed' 
                  ? 'Video processing failed'
                  : 'Video not ready yet'}
              </p>
            </div>
          )}
        </div>

        {/* Job Info */}
        <div className="space-y-6">
          {/* Details */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-4">Details</h2>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-zinc-400">Twitch Clip</dt>
                <dd>
                  {job.twitch_clip_url ? (
                    <a
                      href={job.twitch_clip_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-twitch-purple hover:underline"
                    >
                      {job.twitch_clip_id?.slice(0, 20)}...
                    </a>
                  ) : (
                    <span className="text-zinc-500">-</span>
                  )}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Requested By</dt>
                <dd>{job.requested_by || 'system'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Source</dt>
                <dd>{job.source}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Preset</dt>
                <dd>{job.render_preset || 'default'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Attempts</dt>
                <dd>{job.attempt_count}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Last Stage</dt>
                <dd>{job.last_stage || '-'}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Created</dt>
                <dd>{new Date(job.created_at).toLocaleString()}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-zinc-400">Updated</dt>
                <dd>{new Date(job.updated_at).toLocaleString()}</dd>
              </div>
            </dl>
          </div>

          {/* Review Status */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-4">Review</h2>
            <dl className="space-y-3 text-sm">
              <div className="flex justify-between">
                <dt className="text-zinc-400">Status</dt>
                <dd>
                  {job.review_status ? (
                    <span className={`status-badge review-${job.review_status}`}>
                      {job.review_status}
                    </span>
                  ) : (
                    <span className="text-zinc-500">Not reviewed</span>
                  )}
                </dd>
              </div>
              {job.review_notes && (
                <div>
                  <dt className="text-zinc-400 mb-1">Notes</dt>
                  <dd className="bg-zinc-800 rounded p-2 text-sm">
                    {job.review_notes}
                  </dd>
                </div>
              )}
              {job.reviewed_at && (
                <div className="flex justify-between">
                  <dt className="text-zinc-400">Reviewed At</dt>
                  <dd>{new Date(job.reviewed_at).toLocaleString()}</dd>
                </div>
              )}
            </dl>
          </div>

          {/* Error */}
          {job.error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <h2 className="text-lg font-semibold text-red-400 mb-2">Error</h2>
              <pre className="text-sm text-red-300 whitespace-pre-wrap">
                {job.error}
              </pre>
            </div>
          )}
        </div>
      </div>

      {/* Transcript */}
      {job.transcript_text && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4">Transcript</h2>
          <p className="text-zinc-300 whitespace-pre-wrap text-sm leading-relaxed">
            {job.transcript_text}
          </p>
        </div>
      )}

      {/* Events Log */}
      {events && events.length > 0 && (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <h2 className="text-lg font-semibold mb-4">Processing Log</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {events.map(event => (
              <div
                key={event.id}
                className={`flex gap-3 text-sm p-2 rounded ${
                  event.level === 'error' 
                    ? 'bg-red-500/10' 
                    : event.level === 'warn'
                    ? 'bg-yellow-500/10'
                    : 'bg-zinc-800/50'
                }`}
              >
                <span className="text-zinc-500 font-mono text-xs">
                  {new Date(event.created_at).toLocaleTimeString()}
                </span>
                <span className={`font-medium ${
                  event.level === 'error' ? 'text-red-400' :
                  event.level === 'warn' ? 'text-yellow-400' :
                  'text-zinc-400'
                }`}>
                  [{event.stage || 'system'}]
                </span>
                <span className="text-zinc-300">{event.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

