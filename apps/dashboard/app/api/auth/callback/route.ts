import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  
  // Check for error
  const error = searchParams.get('error')
  if (error) {
    return NextResponse.redirect(new URL(`/login?error=${encodeURIComponent(error)}`, request.url))
  }
  
  // Get user info from query params (set by API callback)
  const channel_id = searchParams.get('channel_id')
  const twitch_id = searchParams.get('twitch_id')
  const login = searchParams.get('login')
  const display_name = searchParams.get('display_name')
  const profile_image_url = searchParams.get('profile_image_url')
  
  if (!channel_id || !twitch_id || !login || !display_name) {
    return NextResponse.redirect(new URL('/login?error=missing_data', request.url))
  }
  
  // Set session cookie
  cookies().set('stream2short_session', JSON.stringify({
    id: channel_id,
    twitch_id: twitch_id,
    twitch_login: login,
    display_name: display_name,
    profile_image_url: profile_image_url || null,
  }), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7, // 7 days
    path: '/',
  })
  
  return NextResponse.redirect(new URL('/', request.url))
}
