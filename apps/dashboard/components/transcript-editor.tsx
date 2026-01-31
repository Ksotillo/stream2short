'use client'

import { useState, useCallback, useEffect, useMemo } from 'react'
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
  User,
  ChevronDown,
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

// Speaker colors for visual distinction
const SPEAKER_COLORS: Record<string, string> = {
  'SPEAKER_0': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  'SPEAKER_1': 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  'SPEAKER_2': 'bg-green-500/20 text-green-400 border-green-500/30',
  'SPEAKER_3': 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  'SPEAKER_4': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
}

function getSpeakerColor(speaker: string | undefined): string {
  if (!speaker) return 'bg-muted text-muted-foreground'
  return SPEAKER_COLORS[speaker] || 'bg-muted text-muted-foreground'
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
  const [openSpeakerDropdown, setOpenSpeakerDropdown] = useState<number | null>(null)
  
  // Get unique speakers from segments
  const availableSpeakers = useMemo(() => {
    const speakers = new Set<string>()
    segments.forEach(s => {
      if (s.speaker) speakers.add(s.speaker)
    })
    // Always include at least 2 speaker options
    if (speakers.size === 0) {
      speakers.add('SPEAKER_0')
      speakers.add('SPEAKER_1')
    } else if (speakers.size === 1) {
      // Add one more option
      const existing = Array.from(speakers)[0]
      const nextNum = parseInt(existing.replace('SPEAKER_', '')) + 1
      speakers.add(`SPEAKER_${nextNum}`)
    }
    return Array.from(speakers).sort()
  }, [segments])
  
  // Reset segments when initialSegments change
  useEffect(() => {
    if (initialSegments) {
      setSegments(initialSegments)
      setHasChanges(false)
    }
  }, [initialSegments])
  
  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = () => setOpenSpeakerDropdown(null)
    if (openSpeakerDropdown !== null) {
      document.addEventListener('click', handleClickOutside)
      return () => document.removeEventListener('click', handleClickOutside)
    }
  }, [openSpeakerDropdown])
  
  const handleTextChange = useCallback((index: number, newText: string) => {
    setSegments(prev => {
      const updated = [...prev]
      updated[index] = { ...updated[index], text: newText }
      return updated
    })
    setHasChanges(true)
    setMessage(null)
  }, [])
  
  const handleSpeakerChange = useCallback((index: number, newSpeaker: string) => {
    setSegments(prev => {
      const updated = [...prev]
      // Determine if this speaker is the primary one (most common)
      const speakerCounts: Record<string, number> = {}
      prev.forEach(s => {
        if (s.speaker) speakerCounts[s.speaker] = (speakerCounts[s.speaker] || 0) + 1
      })
      const primarySpeaker = Object.entries(speakerCounts).sort((a, b) => b[1] - a[1])[0]?.[0]
      
      updated[index] = { 
        ...updated[index], 
        speaker: newSpeaker,
        is_primary: newSpeaker === primarySpeaker,
      }
      return updated
    })
    setHasChanges(true)
    setMessage(null)
    setOpenSpeakerDropdown(null)
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
  
  // Check if any segment has speaker info
  const hasSpeakers = segments.some(s => s.speaker)
  
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
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base flex flex-wrap items-center gap-2">
              <Pencil className="w-4 h-4" />
              Transcript Editor
              {editedAt && (
                <Badge variant="secondary" className="font-normal text-xs">
                  Edited
                </Badge>
              )}
              {hasChanges && (
                <Badge variant="warning" className="font-normal text-xs">
                  Unsaved
                </Badge>
              )}
            </CardTitle>
            <CardDescription className="text-xs sm:text-sm">
              Edit text and speaker assignments. Save and re-render to apply.
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
                  className="gap-1.5 flex-1 sm:flex-none"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span className="sm:inline">Reset</span>
                </Button>
              )}
              <Button
                variant="default"
                size="sm"
                onClick={handleSave}
                disabled={saving || !hasChanges}
                className="gap-1.5 flex-1 sm:flex-none"
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
              <CheckCircle className="w-4 h-4 shrink-0" />
            ) : (
              <AlertCircle className="w-4 h-4 shrink-0" />
            )}
            <span className="text-xs sm:text-sm">{message.text}</span>
          </div>
        )}
      </CardHeader>
      
      <CardContent className="px-3 sm:px-6">
        <div className="space-y-1 max-h-[60vh] overflow-y-auto">
          {segments.map((segment, index) => (
            <div
              key={index}
              className="flex flex-col sm:flex-row gap-2 sm:gap-3 sm:items-center p-2 sm:p-2 rounded-lg hover:bg-muted/50 transition-colors border-b border-border/50 sm:border-0"
            >
              {/* Mobile: Time + Speaker row */}
              <div className="flex items-center justify-between sm:justify-start gap-2 sm:gap-3 sm:shrink-0">
                {/* Timing */}
                <div className="flex items-center gap-1 text-[10px] sm:text-xs text-muted-foreground font-mono">
                  <Clock className="w-3 h-3 hidden sm:block" />
                  <span>{formatTime(segment.start)}</span>
                  <span className="text-muted-foreground/50">-</span>
                  <span>{formatTime(segment.end)}</span>
                </div>
                
                {/* Speaker selector */}
                {hasSpeakers && canEdit && (
                  <div className="relative">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        setOpenSpeakerDropdown(openSpeakerDropdown === index ? null : index)
                      }}
                      className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] sm:text-xs font-medium border transition-colors ${getSpeakerColor(segment.speaker)}`}
                    >
                      <User className="w-3 h-3" />
                      <span className="hidden sm:inline">{segment.speaker || 'None'}</span>
                      <span className="sm:hidden">{segment.speaker?.replace('SPEAKER_', 'S') || '?'}</span>
                      <ChevronDown className="w-3 h-3" />
                    </button>
                    
                    {/* Dropdown */}
                    {openSpeakerDropdown === index && (
                      <div className="absolute right-0 sm:left-0 top-full mt-1 z-10 bg-popover border rounded-lg shadow-lg py-1 min-w-[120px]">
                        {availableSpeakers.map((speaker) => (
                          <button
                            key={speaker}
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              handleSpeakerChange(index, speaker)
                            }}
                            className={`w-full text-left px-3 py-1.5 text-xs hover:bg-muted transition-colors flex items-center gap-2 ${
                              segment.speaker === speaker ? 'bg-muted' : ''
                            }`}
                          >
                            <div className={`w-2 h-2 rounded-full ${getSpeakerColor(speaker).split(' ')[0]}`} />
                            {speaker}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                
                {/* Read-only speaker badge */}
                {hasSpeakers && !canEdit && segment.speaker && (
                  <Badge 
                    variant={segment.is_primary ? 'default' : 'secondary'} 
                    className="text-[10px]"
                  >
                    {segment.speaker}
                  </Badge>
                )}
              </div>
              
              {/* Text input - full width on mobile */}
              {canEdit ? (
                <input
                  type="text"
                  value={segment.text}
                  onChange={(e) => handleTextChange(index, e.target.value)}
                  className="flex-1 bg-muted/30 sm:bg-transparent border border-border/50 sm:border-0 sm:border-b sm:border-transparent rounded sm:rounded-none px-2 py-2 sm:py-1.5 hover:border-border sm:hover:border-border focus:border-primary sm:focus:border-primary focus:outline-none text-sm transition-colors"
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
            Duration: {formatTime(segments[segments.length - 1]?.end || 0)}
          </span>
        </div>
      </CardContent>
      
      {/* Re-render Prompt Modal */}
      {showRerenderPrompt && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md mx-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <RefreshCw className="w-5 h-5" />
                Re-render Video?
              </CardTitle>
              <CardDescription>
                Your changes have been saved. To see them in the video, 
                you&apos;ll need to re-render the clip.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col-reverse sm:flex-row justify-end gap-2">
                <Button 
                  variant="ghost" 
                  onClick={handleSkipRerender}
                  disabled={rerendering}
                  className="w-full sm:w-auto"
                >
                  Not now
                </Button>
                <Button 
                  variant="default"
                  onClick={handleRerender}
                  disabled={rerendering}
                  className="gap-2 w-full sm:w-auto"
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
