'use client'

import { useState, useCallback, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { updateTranscript, rerenderJob, TranscriptSegment } from '@/lib/api'
import { 
  Save, 
  Loader2, 
  RotateCcw, 
  Clock, 
  AlertCircle,
  CheckCircle,
  Pencil,
  RefreshCw,
  X,
} from 'lucide-react'

interface TranscriptEditorProps {
  jobId: string
  segments: TranscriptSegment[] | null
  transcriptText: string | null
  editedAt: string | null
  canEdit: boolean
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  const ms = Math.floor((seconds % 1) * 10)
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`
}

export function TranscriptEditor({ 
  jobId, 
  segments: initialSegments, 
  transcriptText,
  editedAt,
  canEdit,
}: TranscriptEditorProps) {
  const [segments, setSegments] = useState<TranscriptSegment[]>(initialSegments || [])
  const [saving, setSaving] = useState(false)
  const [rerendering, setRerendering] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [hasChanges, setHasChanges] = useState(false)
  const [showRerenderPrompt, setShowRerenderPrompt] = useState(false)
  
  // Reset segments when initialSegments change
  useEffect(() => {
    if (initialSegments) {
      setSegments(initialSegments)
      setHasChanges(false)
    }
  }, [initialSegments])
  
  const handleTextChange = useCallback((index: number, newText: string) => {
    setSegments(prev => {
      const updated = [...prev]
      updated[index] = { ...updated[index], text: newText }
      return updated
    })
    setHasChanges(true)
    setMessage(null)
  }, [])
  
  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    
    try {
      const result = await updateTranscript(jobId, segments)
      setMessage({ type: 'success', text: result.message })
      setHasChanges(false)
      // Show re-render prompt after successful save
      setShowRerenderPrompt(true)
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Failed to save transcript' 
      })
    } finally {
      setSaving(false)
    }
  }
  
  const handleRerender = async () => {
    setRerendering(true)
    try {
      await rerenderJob(jobId, 'default')
      setShowRerenderPrompt(false)
      setMessage({ type: 'success', text: 'Re-render started! The page will refresh shortly...' })
      setTimeout(() => window.location.reload(), 2000)
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: error instanceof Error ? error.message : 'Failed to start re-render' 
      })
    } finally {
      setRerendering(false)
    }
  }
  
  const handleSkipRerender = () => {
    setShowRerenderPrompt(false)
    setTimeout(() => window.location.reload(), 500)
  }
  
  const handleReset = () => {
    if (initialSegments) {
      setSegments(initialSegments)
      setHasChanges(false)
      setMessage(null)
    }
  }
  
  // If no segments, show the plain text transcript (read-only)
  if (!segments || segments.length === 0) {
    if (transcriptText) {
      return (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              Transcript
              <Badge variant="outline" className="font-normal">Read Only</Badge>
            </CardTitle>
            <CardDescription>
              Segment data not available for editing. Showing plain text.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-relaxed text-muted-foreground">
              {transcriptText}
            </p>
          </CardContent>
        </Card>
      )
    }
    return null
  }
  
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-4">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Pencil className="w-4 h-4" />
              Transcript Editor
              {editedAt && (
                <Badge variant="secondary" className="font-normal text-xs">
                  Edited
                </Badge>
              )}
              {hasChanges && (
                <Badge variant="warning" className="font-normal text-xs">
                  Unsaved changes
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="mt-1">
              Edit subtitle text below. Timing is preserved. Save and re-render to apply changes.
            </CardDescription>
          </div>
          
          {canEdit && (
            <div className="flex gap-2 shrink-0">
              {hasChanges && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  disabled={saving}
                  className="gap-1.5"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset
                </Button>
              )}
              <Button
                variant="default"
                size="sm"
                onClick={handleSave}
                disabled={saving || !hasChanges}
                className="gap-1.5"
              >
                {saving ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Save className="w-3.5 h-3.5" />
                )}
                Save
              </Button>
            </div>
          )}
        </div>
        
        {/* Status message */}
        {message && (
          <div className={`flex items-center gap-2 mt-3 text-sm ${
            message.type === 'success' ? 'text-green-500' : 'text-destructive'
          }`}>
            {message.type === 'success' ? (
              <CheckCircle className="w-4 h-4" />
            ) : (
              <AlertCircle className="w-4 h-4" />
            )}
            {message.text}
          </div>
        )}
      </CardHeader>
      
      <CardContent>
        <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
          {segments.map((segment, index) => (
            <div
              key={index}
              className="flex gap-3 items-start p-2 rounded-lg hover:bg-muted/50 transition-colors group"
            >
              {/* Timing */}
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono shrink-0 pt-2">
                <Clock className="w-3 h-3" />
                <span>{formatTime(segment.start)}</span>
                <span className="text-muted-foreground/50">-</span>
                <span>{formatTime(segment.end)}</span>
              </div>
              
              {/* Speaker indicator */}
              {segment.speaker && (
                <Badge 
                  variant={segment.is_primary ? 'default' : 'secondary'} 
                  className="text-[10px] shrink-0 mt-1.5"
                >
                  {segment.speaker}
                </Badge>
              )}
              
              {/* Editable text */}
              {canEdit ? (
                <input
                  type="text"
                  value={segment.text}
                  onChange={(e) => handleTextChange(index, e.target.value)}
                  className="flex-1 bg-transparent border-0 border-b border-transparent hover:border-border focus:border-primary focus:outline-none text-sm py-1.5 transition-colors"
                  placeholder="Enter text..."
                />
              ) : (
                <span className="flex-1 text-sm py-1.5 text-muted-foreground">
                  {segment.text}
                </span>
              )}
            </div>
          ))}
        </div>
        
        {/* Summary */}
        <div className="mt-4 pt-4 border-t flex items-center justify-between text-xs text-muted-foreground">
          <span>{segments.length} segments</span>
          <span>
            Total duration: {formatTime(segments[segments.length - 1]?.end || 0)}
          </span>
        </div>
      </CardContent>
      
      {/* Re-render Prompt Modal */}
      {showRerenderPrompt && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <RefreshCw className="w-5 h-5" />
                Re-render Video?
              </CardTitle>
              <CardDescription>
                Your transcript changes have been saved. To see them in the video, 
                you&apos;ll need to re-render the clip with the updated subtitles.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-end gap-2">
                <Button 
                  variant="ghost" 
                  onClick={handleSkipRerender}
                  disabled={rerendering}
                >
                  Not now
                </Button>
                <Button 
                  variant="default"
                  onClick={handleRerender}
                  disabled={rerendering}
                  className="gap-2"
                >
                  {rerendering ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <RefreshCw className="w-4 h-4" />
                  )}
                  Re-render now
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </Card>
  )
}
