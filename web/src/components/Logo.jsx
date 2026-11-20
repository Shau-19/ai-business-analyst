// Mark: three forest bars growing inside a coral sun — data that grows.
export function LogoMark({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="15" fill="var(--coral)" />
      <rect x="8" y="17" width="4" height="8" rx="1.5" fill="var(--forest)" />
      <rect x="14" y="12" width="4" height="13" rx="1.5" fill="var(--forest)" />
      <rect x="20" y="7" width="4" height="18" rx="1.5" fill="var(--forest)" />
    </svg>
  );
}

export default function Logo({ size = 28 }) {
  return (
    <span className="logo">
      <LogoMark size={size} />
      <span className="logo-word display">Analyst</span>
    </span>
  );
}
