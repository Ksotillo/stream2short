'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { 
  ArrowLeft, 
  Link2, 
  Loader2, 
  CheckCircle,
  Sparkles,
  Film,
} from 'lucide-react'
import Link from 'next/link'

export default function CreateClipPage() {
  const router = useRouter()
  const [clipUrl, setClipUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setIsLoading(true)
    
    try {
      const res = await fetch('/api/clips/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clip_url: clipUrl }),
      })
      
      const data = await res.json()
      
      if (!res.ok) {
        throw new Error(data.error || 'Failed to process clip')
      }
      
      setSuccess(true)
      
      // Redirect to clips page after a moment
      setTimeout(() => {
        router.push('/clips')
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setIsLoading(false)
    }
  }
  
  return (
    <>
      {/* ==================== MOBILE VERSION ==================== */}
      <div className="lg:hidden min-h-screen px-4 py-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <Link href="/clips">
            <motion.button
              whileTap={{ scale: 0.9 }}
              className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center"
            >
              <ArrowLeft className="w-5 h-5 text-white/70" />
            </motion.button>
          </Link>
          <h1 className="text-xl font-bold text-white">Create Clip</h1>
        </div>
        
        {success ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center py-20"
          >
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: 'spring', stiffness: 200, damping: 10, delay: 0.2 }}
              className="w-20 h-20 rounded-full bg-gradient-to-br from-emerald-500 to-green-500 flex items-center justify-center mb-6"
            >
              <CheckCircle className="w-10 h-10 text-white" />
            </motion.div>
            <h2 className="text-xl font-bold text-white mb-2">Clip Queued!</h2>
            <p className="text-white/60 text-center">
              Your clip is being processed. This may take a minute.
            </p>
          </motion.div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Decorative header */}
            <div className="flex flex-col items-center text-center mb-8">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-violet-400" />
              </div>
              <h2 className="text-lg font-semibold text-white mb-1">
                Process a Twitch Clip
              </h2>
              <p className="text-white/50 text-sm">
                Paste a Twitch clip URL to convert it to a vertical short
              </p>
            </div>
            
            {/* URL Input */}
            <div className="space-y-2">
              <label className="text-sm text-white/70 font-medium">
                Twitch Clip URL
              </label>
              <div className="relative">
                <Link2 className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/40" />
                <input
                  type="url"
                  value={clipUrl}
                  onChange={(e) => setClipUrl(e.target.value)}
                  placeholder="https://clips.twitch.tv/..."
                  className="w-full pl-12 pr-4 py-4 bg-white/5 border border-white/10 rounded-2xl text-white placeholder:text-white/30 focus:outline-none focus:border-violet-500/50 focus:ring-2 focus:ring-violet-500/20 transition-all"
                  required
                />
              </div>
              <p className="text-xs text-white/40">
                Supports clips.twitch.tv and twitch.tv/*/clip/* URLs
              </p>
            </div>
            
            {/* Error message */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-xl bg-red-500/10 border border-red-500/20"
              >
                <p className="text-sm text-red-400">{error}</p>
              </motion.div>
            )}
            
            {/* Submit button */}
            <Button
              type="submit"
              disabled={isLoading || !clipUrl}
              className="w-full h-14 bg-gradient-to-r from-violet-500 to-fuchsia-500 hover:from-violet-600 hover:to-fuchsia-600 text-white rounded-2xl font-semibold text-base disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5 mr-2" />
                  Create Short
                </>
              )}
            </Button>
            
            {/* Help text */}
            <div className="p-4 rounded-xl bg-white/5 border border-white/10">
              <h3 className="text-sm font-medium text-white mb-2">💡 Tip</h3>
              <p className="text-xs text-white/50 leading-relaxed">
                You can also create clips directly from chat using the <code className="text-violet-400">!clip</code> command while streaming.
              </p>
            </div>
          </form>
        )}
      </div>
      
      {/* ==================== DESKTOP VERSION ==================== */}
      <div className="hidden lg:block">
        <div className="max-w-2xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Film className="w-5 h-5 text-primary" />
                Process a Twitch Clip
              </CardTitle>
              <CardDescription>
                Paste a Twitch clip URL to convert it into a vertical short video
              </CardDescription>
            </CardHeader>
            <CardContent>
              {success ? (
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="w-16 h-16 rounded-full bg-emerald-500/20 flex items-center justify-center mb-4">
                    <CheckCircle className="w-8 h-8 text-emerald-500" />
                  </div>
                  <h2 className="text-xl font-semibold mb-2">Clip Queued!</h2>
                  <p className="text-muted-foreground text-center">
                    Your clip is being processed. This may take a minute.
                  </p>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Twitch Clip URL
                    </label>
                    <div className="relative">
                      <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <input
                        type="url"
                        value={clipUrl}
                        onChange={(e) => setClipUrl(e.target.value)}
                        placeholder="https://clips.twitch.tv/..."
                        className="w-full pl-10 pr-4 py-3 bg-secondary rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                        required
                      />
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Supports clips.twitch.tv and twitch.tv/*/clip/* URLs
                    </p>
                  </div>
                  
                  {error && (
                    <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20">
                      <p className="text-sm text-destructive">{error}</p>
                    </div>
                  )}
                  
                  <div className="flex gap-3">
                    <Link href="/clips" className="flex-1">
                      <Button variant="outline" className="w-full">
                        Cancel
                      </Button>
                    </Link>
                    <Button
                      type="submit"
                      disabled={isLoading || !clipUrl}
                      className="flex-1"
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        'Process Clip'
                      )}
                    </Button>
                  </div>
                </form>
              )}
            </CardContent>
          </Card>
          
          {/* Help card */}
          <Card className="mt-4">
            <CardContent className="p-4">
              <div className="flex gap-3">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Sparkles className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-medium mb-1">Pro Tip</h3>
                  <p className="text-sm text-muted-foreground">
                    You can also create clips directly from your Twitch chat using the <code className="text-primary bg-primary/10 px-1 rounded">!clip</code> command while streaming.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  )
}
