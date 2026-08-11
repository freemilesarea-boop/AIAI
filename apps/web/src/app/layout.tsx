import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "LUBER MUSIC AI",
  description: "Generate original music from a prompt, lyrics, and vocal style.",
};

const NAV_LINKS = [
  { href: "/create", label: "Create" },
  { href: "/library", label: "Library" },
  { href: "/settings", label: "Settings" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-zinc-800">
          <nav className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <Link href="/" className="text-lg font-bold tracking-tight">
              LUBER<span className="text-violet-400"> MUSIC AI</span>
            </Link>
            <div className="flex items-center gap-6 text-sm text-zinc-300">
              {NAV_LINKS.map((link) => (
                <Link key={link.href} href={link.href} className="hover:text-white">
                  {link.label}
                </Link>
              ))}
              <Link
                href="/login"
                className="rounded-lg bg-zinc-800 px-3 py-1.5 font-medium hover:bg-zinc-700"
              >
                Log in
              </Link>
            </div>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-10">{children}</main>
      </body>
    </html>
  );
}
