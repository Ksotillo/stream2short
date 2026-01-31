import { NextRequest, NextResponse } from 'next/server'
import { getSession } from '@/lib/auth'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
const API_KEY = process.env.DASHBOARD_API_KEY || ''

export async function POST(request: NextRequest) {
  const session = await getSession()
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    const body = await request.json()
    const { jobId, decision, notes } = body

    if (!jobId || !decision) {
      return NextResponse.json(
        { error: 'Missing jobId or decision' },
        { status: 400 }
      )
    }

    const res = await fetch(`${API_URL}/api/jobs/${jobId}/review`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-dashboard-api-key': API_KEY,
      },
      body: JSON.stringify({ decision, notes }),
    })

    const data = await res.json()

    if (!res.ok) {
      return NextResponse.json(data, { status: res.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error('Review error:', error)
    return NextResponse.json(
      { error: 'Failed to review job' },
      { status: 500 }
    )
  }
}
