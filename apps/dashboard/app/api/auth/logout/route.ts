import { NextRequest, NextResponse } from 'next/server'
import { cookies } from 'next/headers'

export async function POST(request: NextRequest) {
  cookies().delete('stream2short_session')
  return NextResponse.redirect(new URL('/login', request.url))
}

export async function GET(request: NextRequest) {
  cookies().delete('stream2short_session')
  return NextResponse.redirect(new URL('/login', request.url))
}

