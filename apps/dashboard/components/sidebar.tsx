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
    <aside className="hidden lg:flex lg:flex-col lg:w-64 lg:border-r border-border bg-card/50 h-screen sticky top-0 shrink-0">
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
    <header className="lg:hidden flex items-center justify-between h-14 px-4 bg-black/80 backdrop-blur-xl border-b border-white/5 sticky top-0 z-50">
      <Avatar className="h-9 w-9 ring-2 ring-violet-500/50">
        {user.profile_image_url && (
          <AvatarImage src={user.profile_image_url} alt={user.display_name} />
        )}
        <AvatarFallback className="bg-violet-500/20 text-violet-300 text-sm font-semibold">
          {user.display_name[0].toUpperCase()}
        </AvatarFallback>
      </Avatar>
      
      <Link href="/">
        <LogoWithText />
      </Link>
      
      <div className="w-9 h-9 rounded-full bg-white/5 flex items-center justify-center">
        <svg
          className="w-5 h-5 text-white/70"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
          />
        </svg>
      </div>
    </header>
  )
}

