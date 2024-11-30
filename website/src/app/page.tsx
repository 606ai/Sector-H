'use client';

import HeroSection from '@/components/ui/HeroSection';
import FeatureGrid from '@/components/ui/FeatureGrid';
import GlowingCard from '@/components/ui/GlowingCard';
import Link from 'next/link';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <main className="min-h-screen bg-black text-white overflow-hidden">
      {/* Hero Section */}
      <HeroSection />

      {/* Feature Grid */}
      <FeatureGrid />

      {/* Call to Action */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <GlowingCard className="h-full">
              <Link href="/dashboard" className="block h-full">
                <h3 className="text-2xl font-bold mb-4 text-gradient">Get Started</h3>
                <p className="text-gray-400 mb-6">
                  Begin your AI development journey with our comprehensive platform.
                </p>
                <motion.span
                  className="inline-block text-indigo-400"
                  whileHover={{ x: 5 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  Launch Dashboard →
                </motion.span>
              </Link>
            </GlowingCard>

            <GlowingCard className="h-full">
              <Link href="/docs" className="block h-full">
                <h3 className="text-2xl font-bold mb-4 text-gradient">Documentation</h3>
                <p className="text-gray-400 mb-6">
                  Explore our comprehensive guides and API documentation.
                </p>
                <motion.span
                  className="inline-block text-indigo-400"
                  whileHover={{ x: 5 }}
                  transition={{ type: "spring", stiffness: 400 }}
                >
                  View Docs →
                </motion.span>
              </Link>
            </GlowingCard>
          </div>
        </div>
      </section>

      {/* GitHub Link */}
      <section className="py-20 px-4 relative overflow-hidden">
        <div className="absolute inset-0 bg-grid opacity-10"></div>
        <div className="max-w-4xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-6 text-gradient">
              Open Source
            </h2>
            <p className="text-gray-400 mb-8">
              Sector-H is open source and available on GitHub. Join our community and contribute to the future of AI development.
            </p>
            <Link
              href="https://github.com/606ai/Sector-H"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block bg-white/10 hover:bg-white/20 text-white px-8 py-3 rounded-lg transition-all hover-glow"
            >
              View on GitHub
            </Link>
          </motion.div>
        </div>
      </section>
    </main>
  );
}
