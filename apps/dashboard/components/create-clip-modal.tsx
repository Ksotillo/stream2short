'use client'

import { useState } from 'react'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { processClip } from '@/lib/api'
import { Plus, Loader2, CheckCircle, XCircle, Link2, Sparkles } from 'lucide-react'

export function CreateClipButton() {
  const [isOpen, setIsOpen] = useState(false)
  
  return (
    <>
      <Button onClick={() => setIsOpen(true)} className="gap-2">
        <Plus className="w-4 h-4" />
        Create Clip
      </Button>
      
      {isOpen && <CreateClipModal onClose={() => setIsOpen(false)} />}
    </>
  )
}

function CreateClipModal({ onClose }: { onClose: () => void }) {
  const [clipUrl, setClipUrl] = useState('')
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')
  const [jobId, setJobId] = useState<string | null>(null)
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!clipUrl.trim()) {
      setMessage('Please enter a Twitch clip URL')
      setStatus('error')
      return
    }
    
    // Basic URL validation
    if (!clipUrl.includes('twitch.tv') && !clipUrl.includes('clips.twitch.tv')) {
      setMessage('Please enter a valid Twitch clip URL')
      setStatus('error')
      return
    }
    
    setStatus('loading')
    setMessage('')
    
    try {
      const result = await processClip(clipUrl, 'dashboard')
      
      if (result.success) {
        setStatus('success')
        setMessage(result.message || 'Clip queued for processing!')
        setJobId(result.job_id || null)
      } else {
        setStatus('error')
        setMessage(result.error || result.message || 'Failed to process clip')
      }
    } catch (err) {
      setStatus('error')
      setMessage(err instanceof Error ? err.message : 'Failed to process clip')
    }
  }
  
  const handleClose = () => {
    if (status === 'success') {
      // Refresh the page to show the new clip
      window.location.reload()
    } else {
      onClose()
    }
  }
  
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-primary" />
            Create New Clip
          </CardTitle>
          <CardDescription>
            Paste a Twitch clip URL to process it into a vertical short
          </CardDescription>
        </CardHeader>
        <CardContent>
          {status === 'success' ? (
            <div className="text-center py-6">
              <CheckCircle className="w-16 h-16 text-emerald-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold mb-2">Clip Queued!</h3>
              <p className="text-muted-foreground mb-4">{message}</p>
              {jobId && (
                <p className="text-xs text-muted-foreground font-mono mb-4">
                  Job ID: {jobId}
                </p>
              )}
              <Button onClick={handleClose}>
                View Clips
              </Button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">
                  Twitch Clip URL
                </label>
                <div className="relative">
                  <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="text"
                    value={clipUrl}
                    onChange={(e) => setClipUrl(e.target.value)}
                    placeholder="https://clips.twitch.tv/... or https://www.twitch.tv/channel/clip/..."
                    className="w-full bg-secondary border-0 rounded-lg pl-10 pr-4 py-3 text-sm focus:ring-2 focus:ring-primary focus:outline-none"
                    disabled={status === 'loading'}
                    autoFocus
                  />
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Supports clips.twitch.tv and twitch.tv/channel/clip/ URLs
                </p>
              </div>
              
              {status === 'error' && message && (
                <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm">
                  <XCircle className="w-4 h-4 mt-0.5 shrink-0" />
                  <span>{message}</span>
                </div>
              )}
              
              <div className="flex justify-end gap-2 pt-2">
                <Button 
                  type="button" 
                  variant="ghost" 
                  onClick={onClose}
                  disabled={status === 'loading'}
                >
                  Cancel
                </Button>
                <Button 
                  type="submit"
                  disabled={status === 'loading' || !clipUrl.trim()}
                  className="gap-2"
                >
                  {status === 'loading' ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Process Clip
                    </>
                  )}
                </Button>
              </div>
            </form>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

