import { redirect } from 'next/navigation'
import { getSession } from '@/lib/auth'
import { LogoWithText } from '@/components/logo'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
const DASHBOARD_URL = process.env.NEXT_PUBLIC_DASHBOARD_URL || 'http://localhost:3001'

export default async function LoginPage({
  searchParams,
}: {
  searchParams: { error?: string }
}) {
  const session = await getSession()
  
  if (session) {
    redirect('/')
  }
  
  const error = searchParams.error
  
  // Build Twitch OAuth URL
  const twitchAuthUrl = `${API_URL}/auth/twitch/start?redirect_uri=${encodeURIComponent(`${DASHBOARD_URL}/api/auth/callback`)}`
  
  return (
    <div className="min-h-screen gradient-bg flex flex-col items-center justify-center p-4">
      {/* Background decoration */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl" />
      </div>
      
      <div className="relative z-10 w-full max-w-md">
        {/* Logo */}
        <div className="flex justify-center mb-8">
          <LogoWithText className="scale-125" />
        </div>
        
        {/* Login Card */}
        <Card className="glass glow-purple">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Welcome Back</CardTitle>
            <CardDescription>
              Sign in with your Twitch account to manage your clips
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {error && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive text-sm">
                {error === 'not_connected' 
                  ? 'Your Twitch account is not connected. Ask the stream owner to connect first.'
                  : error === 'auth_failed'
                  ? 'Authentication failed. Please try again.'
                  : `Error: ${error}`
                }
              </div>
            )}
            
            <a href={twitchAuthUrl} className="block">
              <Button variant="twitch" className="w-full gap-2" size="lg">
                <TwitchIcon className="w-5 h-5" />
                Continue with Twitch
              </Button>
            </a>
            
            <p className="text-center text-xs text-muted-foreground">
              Only connected streamers can access their dashboard.
              <br />
              Need to connect? Visit the main site first.
            </p>
          </CardContent>
        </Card>
        
        {/* Footer */}
        <p className="text-center text-sm text-muted-foreground mt-8">
          Turn your Twitch clips into viral shorts ✨
        </p>
      </div>
    </div>
  )
}

function TwitchIcon({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
      <path d="M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714Z" />
    </svg>
  )
}

