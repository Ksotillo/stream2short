export function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Phone/vertical video frame */}
      <rect
        x="12"
        y="4"
        width="24"
        height="40"
        rx="4"
        className="fill-primary/20 stroke-primary"
        strokeWidth="2"
      />
      
      {/* Screen area with gradient */}
      <rect
        x="14"
        y="8"
        width="20"
        height="32"
        rx="2"
        className="fill-primary/10"
      />
      
      {/* Play button / clip indicator */}
      <path
        d="M20 19L32 24L20 29V19Z"
        className="fill-primary"
      />
      
      {/* Film strip lines on sides */}
      <rect x="8" y="10" width="2" height="4" rx="0.5" className="fill-primary/60" />
      <rect x="8" y="18" width="2" height="4" rx="0.5" className="fill-primary/60" />
      <rect x="8" y="26" width="2" height="4" rx="0.5" className="fill-primary/60" />
      <rect x="8" y="34" width="2" height="4" rx="0.5" className="fill-primary/60" />
      
      <rect x="38" y="10" width="2" height="4" rx="0.5" className="fill-primary/60" />
      <rect x="38" y="18" width="2" height="4" rx="0.5" className="fill-primary/60" />
      <rect x="38" y="26" width="2" height="4" rx="0.5" className="fill-primary/60" />
      <rect x="38" y="34" width="2" height="4" rx="0.5" className="fill-primary/60" />
      
      {/* Sparkle/magic effect */}
      <circle cx="36" cy="10" r="2" className="fill-yellow-400" />
      <path
        d="M36 6V8M36 12V14M32 10H34M38 10H40"
        className="stroke-yellow-400"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  )
}

export function LogoWithText({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <Logo className="w-9 h-9" />
      <span className="font-bold text-xl tracking-tight">
        Stream<span className="text-primary">2</span>Short
      </span>
    </div>
  )
}

