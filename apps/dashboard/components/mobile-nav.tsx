'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { 
  Home, 
  Film, 
  PlusCircle, 
  Settings, 
  User,
} from 'lucide-react'

const navItems = [
  { href: '/', icon: Home, label: 'Home' },
  { href: '/clips', icon: Film, label: 'Clips' },
  { href: '/clips/create', icon: PlusCircle, label: 'Create', isAction: true },
  { href: '/settings', icon: Settings, label: 'Settings' },
  { href: '/profile', icon: User, label: 'Profile' },
]

export function MobileBottomNav() {
  const pathname = usePathname()
  
  return (
    <nav className="lg:hidden fixed bottom-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-xl border-t border-white/10 safe-area-inset-bottom">
      <div className="flex items-center justify-around h-16 px-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href || 
            (item.href !== '/' && pathname.startsWith(item.href))
          
          return (
            <Link
              key={item.href}
              href={item.href}
              className="relative flex flex-col items-center justify-center flex-1 h-full"
            >
              {item.isAction ? (
                // Special create button
                <motion.div
                  whileTap={{ scale: 0.9 }}
                  className="flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg shadow-violet-500/30"
                >
                  <item.icon className="w-6 h-6 text-white" />
                </motion.div>
              ) : (
                <>
                  <motion.div
                    whileTap={{ scale: 0.9 }}
                    className={cn(
                      'flex flex-col items-center gap-1 transition-colors',
                      isActive ? 'text-white' : 'text-white/50'
                    )}
                  >
                    <item.icon className="w-6 h-6" />
                    <span className="text-[10px] font-medium">{item.label}</span>
                  </motion.div>
                  
                  {/* Active indicator dot */}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute -top-0.5 w-1 h-1 rounded-full bg-violet-400"
                      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    />
                  )}
                </>
              )}
            </Link>
          )
        })}
      </div>
    </nav>
  )
}

