'use client';
import dynamic from 'next/dynamic';
import AnimatedBeaker from '../components/AnimatedBeaker';
import { motion } from 'framer-motion';

// Dynamically import ParticleBackground with no SSR
const ParticleBackground = dynamic(() => import('../components/ParticleBackground'), {
  ssr: false,
});

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
  },
};

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24 relative overflow-hidden">
      {/* Particle Background */}
      <div className="absolute inset-0 -z-10">
        <ParticleBackground />
      </div>

      {/* Background gradients */}
      <div className="absolute inset-0 hero-gradient -z-20"></div>
      <div className="absolute inset-0 -z-20" style={{
        background: 'radial-gradient(circle at 50% 50%, rgba(0, 124, 240, 0.1) 0%, transparent 50%)',
      }}></div>

      {/* Header */}
      <motion.div
        className="z-10 w-full max-w-5xl"
        initial="hidden"
        animate="visible"
        variants={containerVariants}
      >
        <motion.h1
          className="text-4xl md:text-6xl font-bold text-center mb-4"
          variants={itemVariants}
        >
          Welcome to{' '}
          <span className="gradient-text">Sector-H</span>
        </motion.h1>
        <motion.p
          className="text-center text-lg text-gray-400"
          variants={itemVariants}
        >
          Your Advanced AI Development Playground
        </motion.p>
      </motion.div>

      {/* Center Icon */}
      <motion.div
        className="relative flex place-items-center my-16"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ type: "spring", duration: 1.5 }}
      >
        <AnimatedBeaker />
        {/* Decorative circles */}
        <div className="absolute -inset-32 blur-3xl opacity-20" style={{
          background: 'radial-gradient(circle, rgba(0,240,255,0.2) 0%, transparent 70%)',
        }}></div>
      </motion.div>

      {/* Navigation Cards */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-7xl z-10"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {[
          {
            title: 'Jupyter',
            description: 'Access Jupyter notebooks for data science and machine learning.',
            href: '/jupyter',
            external: true
          },
          {
            title: 'API Docs',
            description: 'Explore the API documentation and endpoints.',
            href: '/api-docs',
            external: true
          },
          {
            title: 'Playground',
            description: 'Try out AI models and experiments in real-time.',
            href: '/playground',
            external: false
          },
          {
            title: 'GitHub',
            description: 'View the source code and contribute to the project.',
            href: 'https://github.com/yourusername/sector-h',
            external: true
          }
        ].map((item, index) => (
          <motion.a
            key={item.title}
            href={item.href}
            className="card group p-6 hover:cursor-pointer"
            target={item.external ? "_blank" : undefined}
            rel={item.external ? "noopener noreferrer" : undefined}
            variants={itemVariants}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <h2 className="text-2xl font-semibold mb-3 gradient-text">
              {item.title}{' '}
              <span className="inline-block transition-transform group-hover:translate-x-1">
                →
              </span>
            </h2>
            <p className="text-gray-400">
              {item.description}
            </p>
          </motion.a>
        ))}
      </motion.div>

      {/* Decorative gradient orbs */}
      <div className="absolute top-1/4 -left-32 w-64 h-64 blur-3xl opacity-20 animate-pulse" style={{
        background: 'radial-gradient(circle, rgba(0,102,255,0.2) 0%, transparent 70%)',
      }}></div>
      <div className="absolute bottom-1/4 -right-32 w-64 h-64 blur-3xl opacity-20 animate-pulse" style={{
        background: 'radial-gradient(circle, rgba(0,240,255,0.2) 0%, transparent 70%)',
      }}></div>
    </main>
  );
}
