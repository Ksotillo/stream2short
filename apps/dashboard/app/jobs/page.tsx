import { getJobs, getChannels } from '@/lib/api'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

interface PageProps {
  searchParams: {
    channel?: string
    status?: string
    review_status?: string
    cursor?: string
  }
}

export default async function JobsPage({ searchParams }: PageProps) {
  const { channel, status, review_status, cursor } = searchParams

  let jobs: Awaited<ReturnType<typeof getJobs>>['jobs'] = []
  let pagination = { next_cursor: null as string | null, has_more: false }
  let channels: Awaited<ReturnType<typeof getChannels>>['channels'] = []
  let error = ''

  try {
    const [jobsRes, channelsRes] = await Promise.all([
      getJobs({
        channel_id: channel,
        status,
        review_status,
        limit: 20,
        cursor,
      }),
      getChannels(),
    ])
    jobs = jobsRes.jobs
    pagination = jobsRes.pagination
    channels = channelsRes.channels
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load jobs'
  }

  const currentChannel = channels.find(c => c.id === channel)

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold">Jobs</h1>
          {currentChannel && (
            <p className="text-zinc-400">
              Showing jobs for {currentChannel.display_name || currentChannel.twitch_login}
            </p>
          )}
        </div>

        {/* Filters */}
        <div className="flex flex-wrap gap-2">
          {/* Channel Filter */}
          <select
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm"
            defaultValue={channel || ''}
            onChange={(e) => {
              const url = new URL(window.location.href)
              if (e.target.value) {
                url.searchParams.set('channel', e.target.value)
              } else {
                url.searchParams.delete('channel')
              }
              url.searchParams.delete('cursor')
              window.location.href = url.toString()
            }}
          >
            <option value="">All Channels</option>
            {channels.map(ch => (
              <option key={ch.id} value={ch.id}>
                {ch.display_name || ch.twitch_login}
              </option>
            ))}
          </select>

          {/* Status Filter */}
          <select
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm"
            defaultValue={status || ''}
            onChange={(e) => {
              const url = new URL(window.location.href)
              if (e.target.value) {
                url.searchParams.set('status', e.target.value)
              } else {
                url.searchParams.delete('status')
              }
              url.searchParams.delete('cursor')
              window.location.href = url.toString()
            }}
          >
            <option value="">All Statuses</option>
            <option value="queued">Queued</option>
            <option value="downloading,transcribing,rendering,uploading">Processing</option>
            <option value="ready">Ready</option>
            <option value="failed">Failed</option>
          </select>

          {/* Review Status Filter */}
          <select
            className="bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm"
            defaultValue={review_status || ''}
            onChange={(e) => {
              const url = new URL(window.location.href)
              if (e.target.value) {
                url.searchParams.set('review_status', e.target.value)
              } else {
                url.searchParams.delete('review_status')
              }
              url.searchParams.delete('cursor')
              window.location.href = url.toString()
            }}
          >
            <option value="">All Reviews</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>
        </div>
      </div>

      {error ? (
        <div className="text-center py-12">
          <div className="text-red-400 mb-4">⚠️ {error}</div>
        </div>
      ) : jobs.length === 0 ? (
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">
          <p className="text-zinc-400">No jobs found.</p>
        </div>
      ) : (
        <>
          {/* Jobs Table */}
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-zinc-800/50">
                <tr>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Status</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Clip ID</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Requested By</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Review</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Preset</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Created</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {jobs.map(job => (
                  <tr key={job.id} className="hover:bg-zinc-800/30">
                    <td className="px-4 py-3">
                      <span className={`status-badge status-${job.status}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-col">
                        <span className="font-mono text-sm">
                          {job.twitch_clip_id?.slice(0, 25) || '-'}
                        </span>
                        <span className="text-xs text-zinc-500">
                          {job.id.slice(0, 8)}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-zinc-400">
                      {job.requested_by || 'system'}
                    </td>
                    <td className="px-4 py-3">
                      {job.review_status ? (
                        <span className={`status-badge review-${job.review_status}`}>
                          {job.review_status}
                        </span>
                      ) : (
                        <span className="text-zinc-500">-</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-zinc-400 text-sm">
                      {job.render_preset || 'default'}
                    </td>
                    <td className="px-4 py-3 text-zinc-400 text-sm">
                      {new Date(job.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/jobs/${job.id}`}
                        className="text-twitch-purple hover:underline text-sm"
                      >
                        View →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {pagination.has_more && (
            <div className="flex justify-center">
              <Link
                href={`/jobs?${new URLSearchParams({
                  ...(channel ? { channel } : {}),
                  ...(status ? { status } : {}),
                  ...(review_status ? { review_status } : {}),
                  cursor: pagination.next_cursor!,
                }).toString()}`}
                className="bg-zinc-800 hover:bg-zinc-700 px-4 py-2 rounded-lg text-sm transition"
              >
                Load More
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  )
}

