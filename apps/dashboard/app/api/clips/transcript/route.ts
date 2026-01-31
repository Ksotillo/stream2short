import { NextRequest, NextResponse } from 'next/server'
import { getSession } from '@/lib/auth'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
const API_KEY = process.env.DASHBOARD_API_KEY || ''

export async function PATCH(request: NextRequest) {
  // Verify user is authenticated
  const session = await getSession()
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  try {
    const body = await request.json()
    const { jobId, segments } = body

    if (!jobId || !segments) {
      return NextResponse.json(
        { error: 'Missing jobId or segments' },
        { status: 400 }
      )
    }

    // Proxy request to backend API with the API key
    const res = await fetch(`${API_URL}/api/jobs/${jobId}/transcript`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
        'x-dashboard-api-key': API_KEY,
      },
      body: JSON.stringify({ segments }),
    })

    const data = await res.json()

    if (!res.ok) {
      return NextResponse.json(data, { status: res.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error('Transcript update error:', error)
    return NextResponse.json(
      { error: 'Failed to update transcript' },
      { status: 500 }
    )
  }
}
