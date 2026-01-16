import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Stream2Short Dashboard',
  description: 'Manage and review your Twitch clips',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <nav className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-8">
                <a href="/" className="flex items-center gap-2">
                  <span className="text-2xl">🎬</span>
                  <span className="font-bold text-lg">Stream2Short</span>
                </a>
                <div className="hidden md:flex items-center gap-6">
                  <a href="/jobs" className="text-zinc-400 hover:text-white transition">
                    Jobs
                  </a>
                </div>
              </div>
            </div>
          </div>
        </nav>
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  )
}

