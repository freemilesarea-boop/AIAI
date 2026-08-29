import type { Metadata } from "next";

import { AcquisitionBeacon } from "@/components/AcquisitionBeacon";
import { AuthProvider } from "@/components/auth/AuthProvider";
import { EntitlementProvider } from "@/components/EntitlementProvider";
import { PlayerProvider } from "@/components/player/PlayerProvider";
import { AppShell } from "@/components/shell/AppShell";
import { ToastProvider } from "@/components/ui/Toast";
import "./globals.css";

export const metadata: Metadata = {
  title: "BOORDA 부르다 — AI 음악 생성",
  description: "원하는 분위기를 설명하고 가사를 더하면 완성된 트랙이 나옵니다.",
};

/**
 * The shell, the audio element and the toast region are mounted here,
 * above the router, so navigating between pages never interrupts
 * playback and feedback from an action survives the navigation it
 * triggered.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-screen antialiased">
        {/* Renders nothing; reports the landing URL once per tab. */}
        <AcquisitionBeacon />
        <AuthProvider>
          <EntitlementProvider>
            <PlayerProvider>
            <ToastProvider>
              <AppShell>{children}</AppShell>
            </ToastProvider>
            </PlayerProvider>
          </EntitlementProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
