import { motion } from "framer-motion"

export default function WaveAnimation({ isRecording }) {
  return (
    <div className="absolute flex items-center justify-center">
      {[...Array(3)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-40 h-40 rounded-full border-2 border-teal-400"
          animate={{
            scale: isRecording ? [1, 1.5, 2] : 1,
            opacity: isRecording ? [0.6, 0.3, 0] : 0,
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            delay: i * 0.4,
          }}
        />
      ))}
    </div>
  )
}
