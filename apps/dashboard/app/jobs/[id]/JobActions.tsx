'use client'

import { useState } from 'react'
import { Job, reviewJob, retryJob, rerenderJob } from '@/lib/api'

interface JobActionsProps {
  job: Job
}

export function JobActions({ job }: JobActionsProps) {
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [showReviewModal, setShowReviewModal] = useState(false)
  const [showRerenderModal, setShowRerenderModal] = useState(false)

  const handleReview = async (decision: 'approved' | 'rejected', notes?: string) => {
    setLoading(true)
    setMessage('')
    try {
      const res = await reviewJob(job.id, decision, notes)
      setMessage(res.message)
      setShowReviewModal(false)
      // Reload page to show updated status
      setTimeout(() => window.location.reload(), 1000)
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Failed to review')
    } finally {
      setLoading(false)
    }
  }

  const handleRetry = async (fromStage?: string) => {
    setLoading(true)
    setMessage('')
    try {
      const res = await retryJob(job.id, fromStage)
      setMessage(res.message)
      setTimeout(() => window.location.reload(), 1000)
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Failed to retry')
    } finally {
      setLoading(false)
    }
  }

  const handleRerender = async (preset: string) => {
    setLoading(true)
    setMessage('')
    try {
      const res = await rerenderJob(job.id, preset)
      setMessage(res.message)
      setShowRerenderModal(false)
      setTimeout(() => window.location.reload(), 1000)
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Failed to rerender')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      {/* Review Actions (only for ready jobs) */}
      {job.status === 'ready' && job.review_status !== 'approved' && (
        <>
          <button
            onClick={() => handleReview('approved')}
            disabled={loading}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition"
          >
            ✓ Approve
          </button>
          <button
            onClick={() => setShowReviewModal(true)}
            disabled={loading}
            className="bg-red-600 hover:bg-red-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition"
          >
            ✗ Reject
          </button>
        </>
      )}

      {/* Retry (only for failed jobs) */}
      {job.status === 'failed' && (
        <button
          onClick={() => handleRetry()}
          disabled={loading}
          className="bg-yellow-600 hover:bg-yellow-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition"
        >
          ↻ Retry
        </button>
      )}

      {/* Re-render (only for ready jobs) */}
      {job.status === 'ready' && (
        <button
          onClick={() => setShowRerenderModal(true)}
          disabled={loading}
          className="bg-twitch-purple hover:bg-purple-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition"
        >
          🎬 Re-render
        </button>
      )}

      {/* Status message */}
      {message && (
        <span className="self-center text-sm text-zinc-400">{message}</span>
      )}

      {/* Reject Modal */}
      {showReviewModal && (
        <RejectModal
          onClose={() => setShowReviewModal(false)}
          onSubmit={(notes) => handleReview('rejected', notes)}
          loading={loading}
        />
      )}

      {/* Re-render Modal */}
      {showRerenderModal && (
        <RerenderModal
          onClose={() => setShowRerenderModal(false)}
          onSubmit={handleRerender}
          currentPreset={job.render_preset}
          loading={loading}
        />
      )}
    </div>
  )
}

function RejectModal({
  onClose,
  onSubmit,
  loading,
}: {
  onClose: () => void
  onSubmit: (notes: string) => void
  loading: boolean
}) {
  const [notes, setNotes] = useState('')

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold mb-4">Reject Job</h3>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Rejection reason (optional)"
          className="w-full bg-zinc-800 border border-zinc-700 rounded-lg p-3 text-sm resize-none"
          rows={3}
        />
        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition"
          >
            Cancel
          </button>
          <button
            onClick={() => onSubmit(notes)}
            disabled={loading}
            className="bg-red-600 hover:bg-red-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium transition"
          >
            Reject
          </button>
        </div>
      </div>
    </div>
  )
}

function RerenderModal({
  onClose,
  onSubmit,
  currentPreset,
  loading,
}: {
  onClose: () => void
  onSubmit: (preset: string) => void
  currentPreset: string
  loading: boolean
}) {
  const presets = [
    { id: 'default', name: 'Default', desc: 'Standard style' },
    { id: 'clean', name: 'Clean', desc: 'White text, subtle shadow' },
    { id: 'boxed', name: 'Boxed', desc: 'Rounded background, high contrast' },
    { id: 'minimal', name: 'Minimal', desc: 'Small, unobtrusive' },
    { id: 'bold', name: 'Bold', desc: 'Large text, punchy style' },
  ]

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 w-full max-w-md">
        <h3 className="text-lg font-semibold mb-4">Re-render with Preset</h3>
        <p className="text-sm text-zinc-400 mb-4">
          Current preset: <span className="text-white">{currentPreset}</span>
        </p>
        <div className="space-y-2">
          {presets.map(preset => (
            <button
              key={preset.id}
              onClick={() => onSubmit(preset.id)}
              disabled={loading || preset.id === currentPreset}
              className={`w-full text-left p-3 rounded-lg border transition ${
                preset.id === currentPreset
                  ? 'border-twitch-purple bg-twitch-purple/10 cursor-not-allowed'
                  : 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-800'
              }`}
            >
              <div className="font-medium">{preset.name}</div>
              <div className="text-sm text-zinc-400">{preset.desc}</div>
            </button>
          ))}
        </div>
        <div className="flex justify-end mt-4">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  )
}

