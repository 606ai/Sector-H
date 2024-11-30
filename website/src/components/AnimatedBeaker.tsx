import { motion } from 'framer-motion';
import { BeakerIcon } from '@heroicons/react/24/solid';

export default function AnimatedBeaker() {
    return (
        <div className="relative">
            <motion.div
                className="absolute inset-0 bg-gradient-to-t from-blue-500 to-cyan-300 rounded-full opacity-20"
                animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.2, 0.3, 0.2],
                }}
                transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut",
                }}
            />
            <motion.div
                className="relative"
                animate={{
                    y: [-10, 10, -10],
                    rotate: [-5, 5, -5],
                }}
                transition={{
                    duration: 6,
                    repeat: Infinity,
                    ease: "easeInOut",
                }}
            >
                <BeakerIcon className="h-32 w-32 text-blue-500 glow-effect" />
            </motion.div>
            {/* Liquid bubbles */}
            {[...Array(5)].map((_, i) => (
                <motion.div
                    key={i}
                    className="absolute bottom-8 left-1/2 w-2 h-2 bg-blue-400 rounded-full"
                    initial={{ y: 0, x: -4 + (i * 2), opacity: 0 }}
                    animate={{
                        y: [-20 - (i * 10), 0],
                        opacity: [0.8, 0],
                    }}
                    transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: i * 0.4,
                        ease: "easeOut",
                    }}
                />
            ))}
        </div>
    );
}
