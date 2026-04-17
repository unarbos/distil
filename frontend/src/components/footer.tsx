import { API_SETTINGS, TEACHER } from "@/lib/subnet";
import { ThemeToggle } from "@/components/auto-refresh";

export function Footer() {
  return (
    <footer className="border-t border-border/30 py-2 mt-auto">
      <div className="mx-auto max-w-7xl px-3 sm:px-4 flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground font-mono">
        <span>Distil · SN97 · Bittensor</span>
        <div className="flex items-center gap-4 flex-wrap">
          <a href="https://chat.arbos.life" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
            Chat ↗
          </a>
          <a href="/docs" className="hover:text-foreground transition-colors">
            Docs
          </a>
          <a href="https://x.com/arbos_born" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
            X ↗
          </a>
          <a href="https://github.com/unarbos/distil" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
            GitHub ↗
          </a>
          <a href={`https://huggingface.co/${TEACHER.model}`} target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
            Teacher ↗
          </a>
          <a href={`${API_SETTINGS.publicUrl}/docs`} target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
            API ↗
          </a>
          <a href="https://taomarketcap.com/subnets/97" target="_blank" rel="noopener noreferrer" className="hover:text-foreground transition-colors">
            TaoMarketCap ↗
          </a>
          <ThemeToggle />
        </div>
      </div>
    </footer>
  );
}
