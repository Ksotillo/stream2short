'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { reviewJob, retryJob, rerenderJob } from '@/lib/api'
import { CheckCircle, XCircle, RefreshCw, Palette, Loader2 } from 'lucide-react'

interface ClipActionsProps {
  job: {
    id: string
    status: string
    review_status: string | null
    render_preset: string
  }
}

export function ClipActions({ job }: ClipActionsProps) {
  const [loading, setLoading] = useState<string | null>(null)
  const [message, setMessage] = useState('')
  const [showRerenderModal, setShowRerenderModal] = useState(false)
  const [showRejectModal, setShowRejectModal] = useState(false)
  
  const handleAction = async (
    action: () => Promise<{ success: boolean; message: string }>,
    actionName: string
  ) => {
    setLoading(actionName)
    setMessage('')
    try {
      const res = await action()
      setMessage(res.message)
      setTimeout(() => window.location.reload(), 1500)
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Action failed')
    } finally {
      setLoading(null)
    }
  }
  
  const canReview = job.status === 'ready' && job.review_status !== 'approved'
  const canRetry = job.status === 'failed'
  const canRerender = job.status === 'ready'
  
  if (!canReview && !canRetry && !canRerender) {
    return null
  }
  
  return (
    <div className="flex flex-wrap gap-2">
      {/* Approve */}
      {canReview && (
        <Button
          variant="success"
          className="gap-2"
          onClick={() => handleAction(
            () => reviewJob(job.id, 'approved'),
            'approve'
          )}
          disabled={loading !== null}
        >
          {loading === 'approve' ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <CheckCircle className="w-4 h-4" />
          )}
          Approve
        </Button>
      )}
      
      {/* Reject */}
      {canReview && (
        <Button
          variant="destructive"
          className="gap-2"
          onClick={() => setShowRejectModal(true)}
          disabled={loading !== null}
        >
          <XCircle className="w-4 h-4" />
          Reject
        </Button>
      )}
      
      {/* Retry */}
      {canRetry && (
        <Button
          variant="outline"
          className="gap-2"
          onClick={() => handleAction(
            () => retryJob(job.id),
            'retry'
          )}
          disabled={loading !== null}
        >
          {loading === 'retry' ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
          Retry
        </Button>
      )}
      
      {/* Re-render */}
      {canRerender && (
        <Button
          variant="outline"
          className="gap-2"
          onClick={() => setShowRerenderModal(true)}
          disabled={loading !== null}
        >
          <Palette className="w-4 h-4" />
          Re-render
        </Button>
      )}
      
      {/* Status message */}
      {message && (
        <span className="self-center text-sm text-muted-foreground">{message}</span>
      )}
      
      {/* Reject Modal */}
      {showRejectModal && (
        <RejectModal
          onClose={() => setShowRejectModal(false)}
          onSubmit={async (notes) => {
            await handleAction(
              () => reviewJob(job.id, 'rejected', notes),
              'reject'
            )
            setShowRejectModal(false)
          }}
          loading={loading === 'reject'}
        />
      )}
      
      {/* Re-render Modal */}
      {showRerenderModal && (
        <RerenderModal
          onClose={() => setShowRerenderModal(false)}
          onSubmit={async (preset) => {
            await handleAction(
              () => rerenderJob(job.id, preset),
              'rerender'
            )
            setShowRerenderModal(false)
          }}
          currentPreset={job.render_preset}
          loading={loading === 'rerender'}
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
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Reject Clip</CardTitle>
          <CardDescription>
            Add a note explaining why this clip is being rejected
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Rejection reason (optional)"
            className="w-full min-h-[100px] bg-secondary border-0 rounded-lg p-3 text-sm resize-none focus:ring-2 focus:ring-primary focus:outline-none"
          />
          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose} disabled={loading}>
              Cancel
            </Button>
            <Button 
              variant="destructive" 
              onClick={() => onSubmit(notes)}
              disabled={loading}
              className="gap-2"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              Reject
            </Button>
          </div>
        </CardContent>
      </Card>
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
    { id: 'default', name: 'Default', desc: 'Standard TikTok style' },
    { id: 'clean', name: 'Clean', desc: 'White text, subtle shadow' },
    { id: 'boxed', name: 'Boxed', desc: 'Background box, high contrast' },
    { id: 'minimal', name: 'Minimal', desc: 'Small, unobtrusive text' },
    { id: 'bold', name: 'Bold', desc: 'Large text, punchy style' },
  ]
  
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Re-render Clip</CardTitle>
          <CardDescription>
            Choose a different subtitle preset. Current: <span className="text-foreground font-medium">{currentPreset}</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          {presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => preset.id !== currentPreset && onSubmit(preset.id)}
              disabled={loading || preset.id === currentPreset}
              className={`w-full text-left p-4 rounded-lg border transition-colors ${
                preset.id === currentPreset
                  ? 'border-primary bg-primary/5 cursor-not-allowed opacity-50'
                  : 'border-border hover:border-primary/50 hover:bg-secondary'
              }`}
            >
              <div className="font-medium">{preset.name}</div>
              <div className="text-sm text-muted-foreground">{preset.desc}</div>
            </button>
          ))}
          
          <div className="pt-4 flex justify-end">
            <Button variant="ghost" onClick={onClose} disabled={loading}>
              Cancel
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

