/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  images: {
    domains: ['github.com', 'avatars.githubusercontent.com'],
  },
  env: {
    NEXTAUTH_URL: process.env.NEXTAUTH_URL,
    NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET,
    GITHUB_ID: process.env.OAUTH_GITHUB_ID,
    GITHUB_SECRET: process.env.OAUTH_GITHUB_SECRET,
    GOOGLE_ID: process.env.OAUTH_GOOGLE_ID,
    GOOGLE_SECRET: process.env.OAUTH_GOOGLE_SECRET,
    DISCORD_CLIENT_ID: process.env.OAUTH_DISCORD_ID,
    DISCORD_CLIENT_SECRET: process.env.OAUTH_DISCORD_SECRET,
  },
}

module.exports = nextConfig
