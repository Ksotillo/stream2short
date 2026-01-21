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
    <div className="flex flex-col lg:flex-row h-screen overflow-hidden bg-black lg:bg-background">
      {/* Mobile header */}
      <MobileHeader user={session} />
      
      {/* Desktop sidebar */}
      <Sidebar user={session} />
      
      {/* Main content */}
      <main className="flex-1 overflow-y-auto pb-20 lg:pb-0">
        {/* Mobile: no container padding, Desktop: normal container */}
        <div className="lg:container lg:py-6 lg:px-8">
          {children}
        </div>
      </main>
      
      {/* Mobile bottom navigation */}
      <MobileBottomNav />
    </div>
  )
}

