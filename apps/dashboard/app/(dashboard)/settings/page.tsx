import { getSession } from '@/lib/auth'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import { ExternalLink, Twitch, Settings2, Zap } from 'lucide-react'

export const dynamic = 'force-dynamic'

export default async function SettingsPage() {
  const session = await getSession()
  if (!session) return null
  
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Manage your Stream2Short configuration
        </p>
      </div>
      
      {/* Profile */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Connected Account</CardTitle>
          <CardDescription>
            Your Twitch account linked to Stream2Short
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16">
              {session.profile_image_url && (
                <AvatarImage src={session.profile_image_url} alt={session.display_name} />
              )}
              <AvatarFallback className="bg-twitch/20 text-twitch text-xl">
                {session.display_name[0].toUpperCase()}
              </AvatarFallback>
            </Avatar>
            <div className="flex-1">
              <h3 className="text-lg font-semibold">{session.display_name}</h3>
              <p className="text-sm text-muted-foreground">@{session.twitch_login}</p>
            </div>
            <a
              href={`https://twitch.tv/${session.twitch_login}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Button variant="outline" className="gap-2">
                <Twitch className="w-4 h-4" />
                View Channel
                <ExternalLink className="w-3 h-3" />
              </Button>
            </a>
          </div>
        </CardContent>
      </Card>
      
      {/* StreamElements Setup */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Zap className="w-4 h-4 text-primary" />
            StreamElements Integration
          </CardTitle>
          <CardDescription>
            Set up the !clip command in your chat
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            To enable the !clip command in your Twitch chat, add a custom command in StreamElements:
          </p>
          
          <div className="space-y-3">
            <div>
              <p className="text-xs text-muted-foreground mb-1">Command</p>
              <code className="block bg-secondary p-3 rounded-lg text-sm font-mono">
                !clip
              </code>
            </div>
            
            <div>
              <p className="text-xs text-muted-foreground mb-1">Response</p>
              <code className="block bg-secondary p-3 rounded-lg text-sm font-mono break-all">
                {`$(urlfetch ${process.env.NEXT_PUBLIC_API_URL || 'YOUR_API_URL'}/se/clip?channel=$(channel)&user=$(user))`}
              </code>
            </div>
          </div>
          
          <a
            href="https://streamelements.com/dashboard/bot/commands"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block"
          >
            <Button variant="outline" className="gap-2">
              Open StreamElements
              <ExternalLink className="w-3 h-3" />
            </Button>
          </a>
        </CardContent>
      </Card>
      
      {/* Feature Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-primary" />
            Channel Settings
          </CardTitle>
          <CardDescription>
            Configure how clips are processed
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg">
            <div>
              <p className="font-medium">Speaker Diarization</p>
              <p className="text-sm text-muted-foreground">
                Color subtitles by speaker (requires more processing time)
              </p>
            </div>
            <Badge variant="outline">Coming soon</Badge>
          </div>
          
          <div className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg">
            <div>
              <p className="font-medium">Auto-Webcam Detection</p>
              <p className="text-sm text-muted-foreground">
                Automatically detect and crop webcam in clips
              </p>
            </div>
            <Badge variant="success">Enabled</Badge>
          </div>
          
          <div className="flex items-center justify-between p-4 bg-secondary/50 rounded-lg">
            <div>
              <p className="font-medium">Dual Output</p>
              <p className="text-sm text-muted-foreground">
                Generate both subtitled and clean versions
              </p>
            </div>
            <Badge variant="success">Enabled</Badge>
          </div>
        </CardContent>
      </Card>
      
      {/* Danger Zone */}
      <Card className="border-destructive/30">
        <CardHeader>
          <CardTitle className="text-base text-destructive">Danger Zone</CardTitle>
          <CardDescription>
            Irreversible actions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <a href="/api/auth/logout">
            <Button variant="destructive">
              Disconnect Account
            </Button>
          </a>
        </CardContent>
      </Card>
    </div>
  )
}

