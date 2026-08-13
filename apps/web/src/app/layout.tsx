import type { Metadata } from "next";

import { PlayerProvider } from "@/components/player/PlayerProvider";
import { AppShell } from "@/components/shell/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "LUBER — AI music generation",
  description: "Write a brief, add lyrics, and generate a finished track.",
};

/**
 * The shell and the audio element are mounted here, above the router,
 * so navigating between pages never interrupts playback.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <PlayerProvider>
          <AppShell>{children}</AppShell>
        </PlayerProvider>
      </body>
    </html>
  );
}
