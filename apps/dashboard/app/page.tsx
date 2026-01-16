import { getChannels, getJobs } from '@/lib/api'
import Link from 'next/link'

export const dynamic = 'force-dynamic'

export default async function Home() {
  let channels: Awaited<ReturnType<typeof getChannels>>['channels'] = []
  let recentJobs: Awaited<ReturnType<typeof getJobs>>['jobs'] = []
  let error = ''

  try {
    const [channelsRes, jobsRes] = await Promise.all([
      getChannels(),
      getJobs({ limit: 5 }),
    ])
    channels = channelsRes.channels
    recentJobs = jobsRes.jobs
  } catch (e) {
    error = e instanceof Error ? e.message : 'Failed to load data'
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-red-400 mb-4">⚠️ {error}</div>
        <p className="text-zinc-500">
          Make sure the API is running and DASHBOARD_API_KEY is configured.
        </p>
      </div>
    )
  }

  const stats = {
    total: recentJobs.length,
    ready: recentJobs.filter(j => j.status === 'ready').length,
    pending: recentJobs.filter(j => j.review_status === 'pending').length,
    failed: recentJobs.filter(j => j.status === 'failed').length,
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-zinc-400">Manage and review your Twitch clips</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-zinc-400 text-sm mb-1">Connected Channels</div>
          <div className="text-2xl font-bold">{channels.length}</div>
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-zinc-400 text-sm mb-1">Completed</div>
          <div className="text-2xl font-bold text-green-400">{stats.ready}</div>
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-zinc-400 text-sm mb-1">Pending Review</div>
          <div className="text-2xl font-bold text-yellow-400">{stats.pending}</div>
        </div>
        <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-4">
          <div className="text-zinc-400 text-sm mb-1">Failed</div>
          <div className="text-2xl font-bold text-red-400">{stats.failed}</div>
        </div>
      </div>

      {/* Connected Channels */}
      <section>
        <h2 className="text-xl font-semibold mb-4">Connected Channels</h2>
        {channels.length === 0 ? (
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">
            <p className="text-zinc-400">No channels connected yet.</p>
            <p className="text-zinc-500 text-sm mt-2">
              Connect a Twitch channel via the API OAuth flow.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {channels.map(channel => (
              <Link
                key={channel.id}
                href={`/jobs?channel=${channel.id}`}
                className="bg-zinc-900 border border-zinc-800 rounded-lg p-4 hover:border-twitch-purple transition group"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-twitch-purple/20 rounded-full flex items-center justify-center text-twitch-purple font-bold">
                    {(channel.display_name || channel.twitch_login || '?')[0].toUpperCase()}
                  </div>
                  <div>
                    <div className="font-medium group-hover:text-twitch-purple transition">
                      {channel.display_name || channel.twitch_login}
                    </div>
                    <div className="text-sm text-zinc-500">
                      @{channel.twitch_login}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>

      {/* Recent Jobs */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Recent Jobs</h2>
          <Link
            href="/jobs"
            className="text-sm text-twitch-purple hover:underline"
          >
            View all →
          </Link>
        </div>
        {recentJobs.length === 0 ? (
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-8 text-center">
            <p className="text-zinc-400">No jobs yet.</p>
          </div>
        ) : (
          <div className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-zinc-800/50">
                <tr>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Status</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Clip</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Requested By</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Review</th>
                  <th className="text-left px-4 py-3 text-sm font-medium text-zinc-400">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800">
                {recentJobs.map(job => (
                  <tr key={job.id} className="hover:bg-zinc-800/30">
                    <td className="px-4 py-3">
                      <span className={`status-badge status-${job.status}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/jobs/${job.id}`}
                        className="text-twitch-purple hover:underline"
                      >
                        {job.twitch_clip_id?.slice(0, 20) || job.id.slice(0, 8)}...
                      </Link>
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
                      {new Date(job.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

