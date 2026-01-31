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
    const { jobId, preset } = body

    if (!jobId || !preset) {
      return NextResponse.json(
        { error: 'Missing jobId or preset' },
        { status: 400 }
      )
    }

    const res = await fetch(`${API_URL}/api/jobs/${jobId}/rerender`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-dashboard-api-key': API_KEY,
      },
      body: JSON.stringify({ preset }),
    })

    const data = await res.json()

    if (!res.ok) {
      return NextResponse.json(data, { status: res.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error('Rerender error:', error)
    return NextResponse.json(
      { error: 'Failed to rerender job' },
      { status: 500 }
    )
  }
}
