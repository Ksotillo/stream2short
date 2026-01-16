import { redirect } from 'next/navigation'
import { getSession } from '@/lib/auth'
import { Sidebar } from '@/components/sidebar'

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
    <div className="flex min-h-screen">
      <Sidebar user={session} />
      <main className="flex-1 overflow-auto">
        <div className="container py-6 lg:py-8">
          {children}
        </div>
      </main>
    </div>
  )
}

