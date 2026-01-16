import { cookies } from 'next/headers'

const SESSION_COOKIE = 'stream2short_session'

export interface User {
  id: string
  twitch_id: string
  twitch_login: string
  display_name: string
  profile_image_url?: string
}

export async function getSession(): Promise<User | null> {
  const cookieStore = cookies()
  const sessionCookie = cookieStore.get(SESSION_COOKIE)
  
  if (!sessionCookie?.value) {
    return null
  }
  
  try {
    return JSON.parse(sessionCookie.value) as User
  } catch {
    return null
  }
}

export function setSession(user: User): void {
  cookies().set(SESSION_COOKIE, JSON.stringify(user), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24 * 7, // 7 days
    path: '/',
  })
}

export function clearSession(): void {
  cookies().delete(SESSION_COOKIE)
}

