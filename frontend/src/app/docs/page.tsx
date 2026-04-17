import { DocsTab } from "@/components/docs-tab";
import { API_SETTINGS } from "@/lib/subnet";
import { Footer } from "@/components/footer";

async function fetchKingKl(): Promise<number | null> {
  try {
    const res = await fetch(`${API_SETTINGS.publicUrl}/api/h2h/latest`, { cache: "no-store" });
    if (!res.ok) return null;
    const json = await res.json();
    const v = json?.king_h2h_kl;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  } catch {
    return null;
  }
}

export const revalidate = 30;

export default async function DocsPage() {
  const kingKl = await fetchKingKl();
  const scoreToBeat = kingKl != null ? kingKl * 0.99 : null;

  return (
    <div className="min-h-screen flex flex-col">
      <main className="flex-1 mx-auto max-w-5xl w-full px-4 py-8 space-y-6">
        <header className="flex items-baseline justify-between">
          <h1 className="text-2xl font-semibold tracking-tight">Docs</h1>
          <a href="/" className="text-xs text-muted-foreground hover:text-foreground">
            ← Dashboard
          </a>
        </header>
        <DocsTab scoreToBeat={scoreToBeat} kingKl={kingKl} />
      </main>
      <Footer />
    </div>
  );
}
