import type { Metadata } from "next";

import { AuthProvider } from "@/components/auth/AuthProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { AppShell } from "@/components/shell/AppShell";
import { ToastProvider } from "@/components/ui/Toast";
import "./globals.css";

export const metadata: Metadata = {
  title: "LUBER — AI music generation",
  description: "Write a brief, add lyrics, and generate a finished track.",
};

/**
 * The shell, the audio element and the toast region are mounted here,
 * above the router, so navigating between pages never interrupts
 * playback and feedback from an action survives the navigation it
 * triggered.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <AuthProvider>
          <PlayerProvider>
            <ToastProvider>
              <AppShell>{children}</AppShell>
            </ToastProvider>
          </PlayerProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
