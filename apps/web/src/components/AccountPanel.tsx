"use client";

/**
 * The account controls in Settings: display name, password, closing.
 *
 * Two of these are destructive in different ways, and the UI treats them
 * differently on purpose. Changing a password signs every other browser
 * out — that is the point of it, so the form says so before you press
 * anything. Closing an account cannot be undone at all, so it takes two
 * deliberate steps and a password, and the button that does it is the
 * only red one on the page.
 *
 * Nothing here decides anything. The server owns every rule — which
 * password is acceptable, whether a subscription blocks closing, whose
 * account this is. These forms exist to say what will happen before it
 * does, and to report what the server said afterwards.
 */

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { useAuth } from "@/components/auth/AuthProvider";
import { Button, Card } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { changePassword, deleteAccount, updateDisplayName } from "@/lib/api";

type Status = { kind: "idle" } | { kind: "ok"; message: string } | { kind: "error"; message: string };

function Feedback({ status }: { status: Status }) {
  if (status.kind === "idle") return null;
  const error = status.kind === "error";
  return (
    <p
      role={error ? "alert" : "status"}
      className={`text-sm ${error ? "text-[var(--danger)]" : "text-[var(--success)]"}`}
    >
      {status.message}
    </p>
  );
}

function message(error: unknown, fallback: string): string {
  return error instanceof ApiError && error.message ? error.message : fallback;
}

// ── display name ─────────────────────────────────────────────────────

export function DisplayNameForm() {
  const { user, adopt } = useAuth();
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const submitting = useRef(false);

  useEffect(() => setValue(user?.display_name ?? ""), [user?.display_name]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (submitting.current) return;
    submitting.current = true;
    setBusy(true);
    setStatus({ kind: "idle" });
    try {
      adopt(await updateDisplayName(value.trim() || null));
      setStatus({ kind: "ok", message: "표시 이름을 저장했습니다." });
    } catch (caught) {
      setStatus({ kind: "error", message: message(caught, "저장하지 못했습니다.") });
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-2">
      <label htmlFor="display-name" className="text-sm text-[var(--text-secondary)]">
        표시 이름
      </label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          id="display-name"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          maxLength={120}
          disabled={busy}
          placeholder="설정하지 않음"
          className="flex-1 rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)]"
        />
        <Button type="submit" variant="secondary" disabled={busy}>
          {busy ? "저장 중…" : "저장"}
        </Button>
      </div>
      <Feedback status={status} />
    </form>
  );
}

// ── password ─────────────────────────────────────────────────────────

export function PasswordForm() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const submitting = useRef(false);

  const mismatch = confirm.length > 0 && next !== confirm;
  const ready = current.length > 0 && next.length > 0 && confirm.length > 0 && !mismatch;

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (submitting.current || !ready) return;
    submitting.current = true;
    setBusy(true);
    setStatus({ kind: "idle" });
    try {
      await changePassword(current, next, confirm);
      setCurrent("");
      setNext("");
      setConfirm("");
      setStatus({
        kind: "ok",
        message: "비밀번호를 변경했습니다. 다른 기기에서는 다시 로그인해야 합니다.",
      });
    } catch (caught) {
      setStatus({ kind: "error", message: message(caught, "비밀번호를 변경하지 못했습니다.") });
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  }

  const field =
    "rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)]";

  return (
    <form onSubmit={submit} className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <label htmlFor="current-password" className="text-sm text-[var(--text-secondary)]">
          현재 비밀번호
        </label>
        <input
          id="current-password"
          type="password"
          autoComplete="current-password"
          value={current}
          onChange={(event) => setCurrent(event.target.value)}
          disabled={busy}
          className={field}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <label htmlFor="new-password" className="text-sm text-[var(--text-secondary)]">
          새 비밀번호
        </label>
        <input
          id="new-password"
          type="password"
          autoComplete="new-password"
          value={next}
          onChange={(event) => setNext(event.target.value)}
          disabled={busy}
          className={field}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <label htmlFor="confirm-password" className="text-sm text-[var(--text-secondary)]">
          새 비밀번호 확인
        </label>
        <input
          id="confirm-password"
          type="password"
          autoComplete="new-password"
          value={confirm}
          onChange={(event) => setConfirm(event.target.value)}
          disabled={busy}
          aria-invalid={mismatch}
          aria-describedby={mismatch ? "confirm-mismatch" : undefined}
          className={field}
        />
        {mismatch ? (
          <p id="confirm-mismatch" className="text-xs text-[var(--danger)]">
            새 비밀번호가 일치하지 않습니다.
          </p>
        ) : null}
      </div>

      <p className="text-xs text-[var(--text-muted)]">
        비밀번호를 바꾸면 이 브라우저를 제외한 모든 기기에서 로그아웃됩니다.
      </p>

      <div className="flex items-center gap-3">
        <Button type="submit" variant="secondary" disabled={busy || !ready}>
          {busy ? "변경 중…" : "비밀번호 변경"}
        </Button>
        <Feedback status={status} />
      </div>
    </form>
  );
}

// ── closing the account ──────────────────────────────────────────────

/**
 * Two steps, and the second one asks for a password.
 *
 * The first step is what happens; the second is proof the person at the
 * keyboard is the account holder. A session cookie left open on a shared
 * machine should not be enough to close somebody's account.
 */
function DeleteDialog({ onClose }: { onClose: () => void }) {
  const [step, setStep] = useState<1 | 2>(1);
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submitting = useRef(false);
  const router = useRouter();
  const firstButton = useRef<HTMLButtonElement>(null);

  useEffect(() => firstButton.current?.focus(), [step]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [busy, onClose]);

  async function confirm() {
    if (submitting.current || password.length === 0) return;
    submitting.current = true;
    setBusy(true);
    setError(null);
    try {
      await deleteAccount(password);
      // Hard navigation: every cached authenticated view has to go, and
      // the session behind them no longer exists.
      window.location.assign("/login");
    } catch (caught) {
      const code = caught instanceof ApiError ? caught.code : undefined;
      setError(
        code === "SUBSCRIPTION_ACTIVE"
          ? "구독이 진행 중입니다. 먼저 구독을 해지한 뒤 탈퇴해 주세요. 계정만 삭제되면 결제가 계속될 수 있습니다."
          : message(caught, "탈퇴를 처리하지 못했습니다."),
      );
      submitting.current = false;
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <button
        type="button"
        aria-label="닫기"
        onClick={() => !busy && onClose()}
        className="absolute inset-0 bg-black/60"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-title"
        className="relative w-full max-w-md rounded-[var(--radius-lg)] border border-[var(--border-default)] bg-[var(--surface-raised)] p-6 shadow-xl"
      >
        <h2 id="delete-title" className="text-lg font-semibold text-[var(--text-primary)]">
          {step === 1 ? "회원 탈퇴 안내" : "본인 확인"}
        </h2>

        {step === 1 ? (
          <>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              계정을 삭제하면 BOORDA 계정과 관련 데이터에 더 이상 접근할 수 없습니다. 이 작업은
              되돌릴 수 없습니다.
            </p>
            <ul className="mt-3 flex list-disc flex-col gap-1 pl-5 text-sm text-[var(--text-secondary)]">
              <li>로그인할 수 없게 되며 모든 기기에서 로그아웃됩니다.</li>
              <li>라이브러리와 만든 음악에 접근할 수 없게 됩니다.</li>
              <li>결제 내역은 회계·분쟁 대응을 위해 보관됩니다.</li>
              <li>구독이 진행 중이면 먼저 해지해야 합니다.</li>
            </ul>
            <div className="mt-5 flex items-center justify-end gap-2">
              <Button type="button" variant="secondary" onClick={onClose}>
                취소
              </Button>
              <Button ref={firstButton} type="button" variant="danger" onClick={() => setStep(2)}>
                계속
              </Button>
            </div>
          </>
        ) : (
          <>
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              계속하려면 현재 비밀번호를 입력해 주세요.
            </p>
            <div className="mt-4 flex flex-col gap-1.5">
              <label htmlFor="delete-password" className="text-sm text-[var(--text-secondary)]">
                현재 비밀번호
              </label>
              <input
                id="delete-password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                disabled={busy}
                className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--brand)]"
              />
            </div>

            {error ? (
              <p role="alert" className="mt-3 text-sm text-[var(--danger)]">
                {error}
              </p>
            ) : null}

            <div className="mt-5 flex items-center justify-end gap-2">
              <Button type="button" variant="secondary" onClick={onClose} disabled={busy}>
                취소
              </Button>
              <Button
                ref={firstButton}
                type="button"
                variant="danger"
                onClick={confirm}
                disabled={busy || password.length === 0}
              >
                {busy ? "처리 중…" : "회원 탈퇴"}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function DangerZone() {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Card className="border-[var(--danger)]/40 p-5">
        <h3 className="text-sm font-semibold text-[var(--danger)]">회원 탈퇴</h3>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          계정을 삭제하면 BOORDA 계정과 관련 데이터에 더 이상 접근할 수 없습니다.
        </p>
        <div className="mt-3">
          <Button type="button" variant="danger" onClick={() => setOpen(true)}>
            회원 탈퇴
          </Button>
        </div>
      </Card>
      {open ? <DeleteDialog onClose={() => setOpen(false)} /> : null}
    </>
  );
}

// ── sign out ─────────────────────────────────────────────────────────

export function SignOutButton() {
  const { signOut } = useAuth();
  const [busy, setBusy] = useState(false);
  const submitting = useRef(false);

  const run = useCallback(async () => {
    if (submitting.current) return;
    submitting.current = true;
    setBusy(true);
    try {
      await signOut();
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  }, [signOut]);

  return (
    <Button type="button" variant="secondary" onClick={run} disabled={busy}>
      {busy ? "로그아웃 중…" : "로그아웃"}
    </Button>
  );
}
