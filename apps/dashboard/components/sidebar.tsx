'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { LogoWithText } from './logo'
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar'
import { Button } from './ui/button'
import { cn } from '@/lib/utils'
import type { User } from '@/lib/auth'
import {
  LayoutDashboard,
  Film,
  Settings,
  LogOut,
  ExternalLink,
} from 'lucide-react'

interface SidebarProps {
  user: User
}

const navItems = [
  {
    title: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
  },
  {
    title: 'Clips',
    href: '/clips',
    icon: Film,
  },
  {
    title: 'Settings',
    href: '/settings',
    icon: Settings,
  },
]

export function Sidebar({ user }: SidebarProps) {
  const pathname = usePathname()
  
  return (
    <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:border-r border-border bg-card/50">
      {/* Logo */}
      <div className="flex h-16 items-center gap-2 px-6 border-b border-border">
        <Link href="/">
          <LogoWithText />
        </Link>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1">
        {navItems.map((item) => {
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              )}
            >
              <item.icon className="w-5 h-5" />
              {item.title}
            </Link>
          )
        })}
        
        {/* External link to Twitch */}
        <a
          href={`https://twitch.tv/${user.twitch_login}`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
        >
          <ExternalLink className="w-5 h-5" />
          View Channel
        </a>
      </nav>
      
      {/* User section */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-3 px-2 py-2">
          <Avatar className="h-9 w-9">
            {user.profile_image_url && (
              <AvatarImage src={user.profile_image_url} alt={user.display_name} />
            )}
            <AvatarFallback className="bg-primary/20 text-primary">
              {user.display_name[0].toUpperCase()}
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{user.display_name}</p>
            <p className="text-xs text-muted-foreground truncate">@{user.twitch_login}</p>
          </div>
        </div>
        
        <a href="/api/auth/logout">
          <Button variant="ghost" className="w-full justify-start gap-2 mt-2 text-muted-foreground hover:text-destructive">
            <LogOut className="w-4 h-4" />
            Sign out
          </Button>
        </a>
      </div>
    </aside>
  )
}

// Mobile header for small screens
export function MobileHeader({ user }: SidebarProps) {
  return (
    <header className="lg:hidden flex items-center justify-between h-16 px-4 border-b border-border bg-card/50">
      <Link href="/">
        <LogoWithText />
      </Link>
      <Avatar className="h-8 w-8">
        {user.profile_image_url && (
          <AvatarImage src={user.profile_image_url} alt={user.display_name} />
        )}
        <AvatarFallback className="bg-primary/20 text-primary text-sm">
          {user.display_name[0].toUpperCase()}
        </AvatarFallback>
      </Avatar>
    </header>
  )
}

