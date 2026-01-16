/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    domains: ['static-cdn.jtvnw.net'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'static-cdn.jtvnw.net',
        pathname: '/**',
      },
    ],
  },
}

module.exports = nextConfig
