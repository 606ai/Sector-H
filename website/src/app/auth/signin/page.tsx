'use client';

import { signIn } from 'next-auth/react';
import { motion } from 'framer-motion';
import Image from 'next/image';

const providers = [
  {
    id: 'google',
    name: 'Google',
    icon: '/icons/google.svg',
    color: 'bg-white hover:bg-gray-100',
    textColor: 'text-gray-800',
  },
  {
    id: 'github',
    name: 'GitHub',
    icon: '/icons/github.svg',
    color: 'bg-gray-900 hover:bg-gray-800',
    textColor: 'text-white',
  },
  {
    id: 'discord',
    name: 'Discord',
    icon: '/icons/discord.svg',
    color: 'bg-[#7289DA] hover:bg-[#677BC4]',
    textColor: 'text-white',
  },
];

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

export default function SignIn() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#030014] px-4">
      <div className="absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,124,240,0.1),transparent_50%)]" />
      </div>
      
      <motion.div
        className="max-w-md w-full space-y-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div variants={itemVariants} className="text-center">
          <h2 className="mt-6 text-3xl font-bold tracking-tight text-white">
            Welcome to <span className="gradient-text">Sector-H</span>
          </h2>
          <p className="mt-2 text-sm text-gray-400">
            Sign in to access your AI development playground
          </p>
        </motion.div>

        <motion.div variants={itemVariants} className="mt-8 space-y-4">
          {providers.map((provider) => (
            <motion.button
              key={provider.id}
              onClick={() => signIn(provider.id, { callbackUrl: '/' })}
              className={`w-full flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-lg ${provider.color} ${provider.textColor} transition-all duration-200 transform hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <Image
                src={provider.icon}
                alt={provider.name}
                width={24}
                height={24}
                className="mr-3"
              />
              Sign in with {provider.name}
            </motion.button>
          ))}
        </motion.div>

        <motion.div
          variants={itemVariants}
          className="mt-8 text-center text-sm text-gray-400"
        >
          By signing in, you agree to our{' '}
          <a href="/terms" className="font-medium text-blue-400 hover:text-blue-300">
            Terms of Service
          </a>{' '}
          and{' '}
          <a href="/privacy" className="font-medium text-blue-400 hover:text-blue-300">
            Privacy Policy
          </a>
        </motion.div>
      </motion.div>
    </div>
  );
}
