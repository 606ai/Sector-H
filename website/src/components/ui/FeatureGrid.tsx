import { motion } from 'framer-motion';
import GlowingCard from './GlowingCard';

const features = [
  {
    title: 'Multi-Agent Systems',
    description: 'Implement and train multiple AI agents in a shared environment',
    icon: '🤖',
  },
  {
    title: 'Causal Learning',
    description: 'Discover and leverage causal relationships in your data',
    icon: '🔄',
  },
  {
    title: 'Meta-Learning',
    description: 'Train models that can quickly adapt to new tasks',
    icon: '🧠',
  },
  {
    title: 'Computer Vision',
    description: 'Advanced image processing and analysis capabilities',
    icon: '👁️',
  },
  {
    title: 'Natural Language Processing',
    description: 'State-of-the-art language processing capabilities',
    icon: '💬',
  },
  {
    title: 'Generative AI',
    description: 'Create and manipulate content with advanced AI models',
    icon: '✨',
  },
];

export default function FeatureGrid() {
  return (
    <div className="py-20 px-4">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="max-w-7xl mx-auto"
      >
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-12 bg-clip-text text-transparent bg-gradient-to-r from-indigo-500 to-purple-500">
          Advanced Features
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <GlowingCard key={feature.title} delay={index * 0.1}>
              <div className="space-y-4">
                <span className="text-4xl">{feature.icon}</span>
                <h3 className="text-xl font-semibold text-white">{feature.title}</h3>
                <p className="text-gray-400">{feature.description}</p>
              </div>
            </GlowingCard>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
