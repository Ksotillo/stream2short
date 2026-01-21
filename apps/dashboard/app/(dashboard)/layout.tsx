import { redirect } from 'next/navigation'
import { getSession } from '@/lib/auth'
import { Sidebar, MobileHeader } from '@/components/sidebar'
import { MobileBottomNav } from '@/components/mobile-nav'

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const session = await getSession()
  
  if (!session) {
    redirect('/login')
  }
  
  return (
    <div className="flex flex-col lg:flex-row h-screen overflow-hidden bg-black">
      {/* Mobile header */}
      <MobileHeader user={session} />
      
      {/* Desktop sidebar */}
      <Sidebar user={session} />
      
      {/* Main content */}
      <main className="flex-1 overflow-y-auto pb-20 lg:pb-0">
        {children}
      </main>
      
      {/* Mobile bottom navigation */}
      <MobileBottomNav />
    </div>
  )
}

