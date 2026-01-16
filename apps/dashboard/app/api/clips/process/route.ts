import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000'
const API_KEY = process.env.DASHBOARD_API_KEY || ''

export async function POST(request: NextRequest) {
  // Check session
  const session = cookies().get('stream2short_session')
  if (!session?.value) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }
  
  try {
    const body = await request.json()
    const { clip_url, requested_by } = body
    
    if (!clip_url) {
      return NextResponse.json({ error: 'clip_url is required' }, { status: 400 })
    }
    
    // Call the API server-side (no mixed content issues)
    const res = await fetch(`${API_URL}/process-clip`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-dashboard-api-key': API_KEY,
      },
      body: JSON.stringify({
        clip_url,
        requested_by: requested_by || 'dashboard',
      }),
    })
    
    const data = await res.json()
    
    if (!res.ok) {
      return NextResponse.json(data, { status: res.status })
    }
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Process clip error:', error)
    return NextResponse.json(
      { error: 'Failed to process clip' },
      { status: 500 }
    )
  }
}

