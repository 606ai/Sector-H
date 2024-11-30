import { motion } from 'framer-motion';
import { ReactNode } from 'react';

interface GlowingCardProps {
  children: ReactNode;
  className?: string;
  delay?: number;
}

export default function GlowingCard({ children, className = '', delay = 0 }: GlowingCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay }}
      className={`relative p-6 rounded-2xl overflow-hidden ${className}`}
    >
      {/* Glow Effect */}
      <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10"></div>
      <div className="absolute inset-[1px] rounded-2xl bg-black/90 backdrop-blur-xl"></div>

      {/* Border */}
      <div className="absolute inset-0 rounded-2xl border border-white/10"></div>

      {/* Content */}
      <div className="relative z-10">{children}</div>
    </motion.div>
  );
}
