import { getSession } from '@/lib/auth'
import { Button } from '@/components/ui/button'
import Link from 'next/link'
import { 
  ExternalLink, 
  LogOut, 
  Settings,
  Film,
  Zap,
  ChevronRight,
} from 'lucide-react'

export const dynamic = 'force-dynamic'

export default async function ProfilePage() {
  const session = await getSession()
  
  if (!session) return null
  
  return (
    <div className="min-h-screen px-4 py-6">
      {/* Profile Header */}
      <div className="flex flex-col items-center text-center mb-8">
        {session.profile_image_url ? (
          <img 
            src={session.profile_image_url} 
            alt={session.display_name}
            className="w-24 h-24 rounded-full object-cover ring-4 ring-violet-500/30 mb-4"
          />
        ) : (
          <div className="w-24 h-24 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-3xl font-bold text-white mb-4">
            {session.display_name[0]}
          </div>
        )}
        
        <h1 className="text-2xl font-bold text-white mb-1">{session.display_name}</h1>
        <p className="text-white/60">@{session.twitch_login}</p>
        
        {/* Twitch link */}
        <a
          href={`https://twitch.tv/${session.twitch_login}`}
          target="_blank"
          rel="noopener noreferrer"
          className="mt-4 inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#9146ff]/20 text-[#9146ff] text-sm font-medium hover:bg-[#9146ff]/30 transition-colors"
        >
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M11.571 4.714h1.715v5.143H11.57zm4.715 0H18v5.143h-1.714zM6 0L1.714 4.286v15.428h5.143V24l4.286-4.286h3.428L22.286 12V0zm14.571 11.143l-3.428 3.428h-3.429l-3 3v-3H6.857V1.714h13.714z"/>
          </svg>
          View Channel
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>
      
      {/* Menu Items */}
      <div className="space-y-2 mb-8">
        <h2 className="text-sm font-medium text-white/40 uppercase tracking-wider mb-3 px-1">
          Quick Links
        </h2>
        
        <MenuItem 
          href="/clips"
          icon={Film}
          label="My Clips"
          description="View and manage your clips"
        />
        
        <MenuItem 
          href="/clips/create"
          icon={Zap}
          label="Create Clip"
          description="Process a new Twitch clip"
        />
        
        <MenuItem 
          href="/settings"
          icon={Settings}
          label="Settings"
          description="Configure your preferences"
        />
      </div>
      
      {/* Logout */}
      <div className="pt-4 border-t border-white/10">
        <a href="/api/auth/logout">
          <Button 
            variant="ghost" 
            className="w-full justify-between text-red-400 hover:text-red-300 hover:bg-red-500/10 h-14"
          >
            <div className="flex items-center gap-3">
              <LogOut className="w-5 h-5" />
              <span>Sign out</span>
            </div>
            <ChevronRight className="w-5 h-5 text-white/20" />
          </Button>
        </a>
      </div>
      
      {/* Footer */}
      <div className="mt-8 text-center">
        <p className="text-xs text-white/30">Stream2Short v1.0</p>
        <p className="text-xs text-white/20 mt-1">Made with 💜 for creators</p>
      </div>
    </div>
  )
}

function MenuItem({
  href,
  icon: Icon,
  label,
  description,
}: {
  href: string
  icon: React.ElementType
  label: string
  description: string
}) {
  return (
    <Link href={href}>
      <div className="flex items-center gap-4 p-4 rounded-2xl bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500/20 to-fuchsia-500/20 flex items-center justify-center">
          <Icon className="w-6 h-6 text-violet-400" />
        </div>
        <div className="flex-1">
          <p className="font-medium text-white">{label}</p>
          <p className="text-sm text-white/50">{description}</p>
        </div>
        <ChevronRight className="w-5 h-5 text-white/20" />
      </div>
    </Link>
  )
}

