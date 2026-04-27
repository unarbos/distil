import type { Metadata } from "next";
import { Inter, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Footer } from "@/components/footer";

// Inter (with the typographic features we set in globals.css) is the
// primary face. Geist_Mono kept for the few places we want monospace
// (commit hashes, model revisions, etc).
const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Distil — SN97",
  description:
    "Bittensor Subnet 97 — Distil. Validators score every miner on a 17-axis composite (math, code, reasoning, IFEval, AIME, MBPP, tool-use, long-context, robustness, on-policy RKL, KL, capability, judge, chat-turns, length, degeneracy, reasoning-density). Composite.worst is the ranking key.",
};

/**
 * Inline no-flash theme script. Runs before any paint to set
 * data-theme on <html> based on saved localStorage value. Without
 * this, dark-mode users see a single-frame flash of light when the
 * page first loads (the React-driven ThemeToggle.apply() doesn't run
 * until after hydration).
 *
 * Equivalent to next-themes' useScript pattern. Uses the same key
 * ("distil:theme") as ThemeToggle.
 */
const noFlashScript = `
(function(){
  try {
    var k = "distil:theme";
    var v = localStorage.getItem(k);
    if (v === "dark" || v === "light") {
      document.documentElement.setAttribute("data-theme", v);
    }
  } catch(_) {}
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${geistMono.variable} h-full antialiased`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: noFlashScript }} />
      </head>
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <main className="flex-1">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
