"use client";

/**
 * Who may run the service, and who may decide that.
 *
 * Only a super administrator reaches this page, and only a super
 * administrator reaches the routes behind it. The separation exists so
 * that a compromised operator account cannot quietly grant itself
 * permanence.
 *
 * Two things this page will not do.
 *
 * It does not create accounts. Promotion is by email address, and the
 * address is a *lookup* — the person must already have registered with
 * a password only they know. An operator who mistypes gets "no such
 * account", not an invitation sent into the void.
 *
 * It does not remove the last super administrator. The server refuses
 * it; this page says why rather than showing a generic failure, because
 * the recovery from an empty console is a hand-run database migration.
 */

import { useCallback, useEffect, useState } from "react";

import { Button, Card, Skeleton, inputClass, labelClass } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";
import { useAuth } from "@/components/auth/AuthProvider";
import { ApiError } from "@/lib/api";
import {
  ROLE_LABELS,
  changeRole,
  fetchAdmins,
  formatDateTime,
  grantAdmin,
  revokeAdmin,
  type AdminUser,
  type UserRole,
} from "@/lib/admin";

export default function AdminAdminsPage() {
  const [admins, setAdmins] = useState<AdminUser[] | null>(null);
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<UserRole>("ADMIN");
  const [busy, setBusy] = useState(false);
  const { notify, notifyError } = useToast();
  const { user } = useAuth();

  const reload = useCallback(async (signal?: AbortSignal) => {
    try {
      setAdmins(await fetchAdmins(signal));
    } catch {
      if (signal?.aborted) return;
      setAdmins([]);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void reload(controller.signal);
    return () => controller.abort();
  }, [reload]);

  const report = (error: unknown, fallback: string) => {
    // A 409 here is the lockout guard, and it deserves its own words:
    // "failed" would leave the operator retrying a refusal that will
    // never succeed.
    if (error instanceof ApiError && error.status === 409) {
      notifyError("최고 관리자는 최소 한 명이 남아 있어야 합니다.");
      return;
    }
    if (error instanceof ApiError && error.status === 404) {
      notifyError("해당 이메일로 가입된 계정이 없습니다.");
      return;
    }
    notifyError(fallback);
  };

  const grant = async () => {
    setBusy(true);
    try {
      await grantAdmin(email.trim().toLowerCase(), role);
      await reload();
      notify("권한을 부여했습니다.");
      setEmail("");
    } catch (error) {
      report(error, "권한 부여에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  const change = async (target: AdminUser, next: UserRole) => {
    setBusy(true);
    try {
      await changeRole(target.id, next);
      await reload();
      notify("등급을 변경했습니다.");
    } catch (error) {
      report(error, "등급 변경에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  const revoke = async (target: AdminUser) => {
    setBusy(true);
    try {
      await revokeAdmin(target.id);
      await reload();
      notify("권한을 해제했습니다.");
    } catch (error) {
      report(error, "권한 해제에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">관리자</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          권한 변경은 모두 활동 기록에 남습니다.
        </p>
      </header>

      <Card className="p-5">
        <form
          className="flex flex-wrap items-end gap-3"
          onSubmit={(event) => {
            event.preventDefault();
            void grant();
          }}
        >
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <label className={labelClass} htmlFor="grant-email">
              이메일
            </label>
            <input
              id="grant-email"
              type="email"
              className={inputClass}
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="이미 가입된 계정의 이메일"
              required
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <label className={labelClass} htmlFor="grant-role">
              등급
            </label>
            <select
              id="grant-role"
              className={inputClass}
              value={role}
              onChange={(event) => setRole(event.target.value as UserRole)}
            >
              <option value="ADMIN">{ROLE_LABELS.ADMIN}</option>
              <option value="SUPER_ADMIN">{ROLE_LABELS.SUPER_ADMIN}</option>
            </select>
          </div>
          <Button type="submit" variant="primary" busy={busy}>
            권한 부여
          </Button>
        </form>
        <p className="mt-3 text-xs text-[var(--text-muted)]">
          이미 가입한 회원만 관리자로 지정할 수 있습니다. 이 화면에서 새 계정은 만들어지지
          않습니다.
        </p>
      </Card>

      {admins === null ? (
        <Skeleton className="h-40 w-full" />
      ) : (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-sm">
            <caption className="sr-only">관리자 목록</caption>
            <thead>
              <tr className="border-b border-[var(--border-default)] text-left text-xs text-[var(--text-muted)]">
                <th scope="col" className="p-3 font-medium">
                  이메일
                </th>
                <th scope="col" className="p-3 font-medium">
                  등급
                </th>
                <th scope="col" className="p-3 font-medium">
                  가입일
                </th>
                <th scope="col" className="p-3 font-medium">
                  <span className="sr-only">권한 변경</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {admins.map((admin) => (
                <tr key={admin.id} className="border-b border-[var(--border-default)] last:border-0">
                  <td className="p-3">
                    <span className="text-[var(--text-primary)]">{admin.email}</span>
                    {admin.id === user?.id ? (
                      <span className="ml-2 text-xs text-[var(--text-muted)]">(나)</span>
                    ) : null}
                  </td>
                  <td className="p-3 text-[var(--text-secondary)]">{ROLE_LABELS[admin.role]}</td>
                  <td className="p-3 text-[var(--text-secondary)]">
                    {formatDateTime(admin.created_at)}
                  </td>
                  <td className="p-3">
                    <div className="flex justify-end gap-2">
                      <Button
                        size="sm"
                        busy={busy}
                        onClick={() =>
                          void change(
                            admin,
                            admin.role === "SUPER_ADMIN" ? "ADMIN" : "SUPER_ADMIN",
                          )
                        }
                      >
                        {admin.role === "SUPER_ADMIN" ? "관리자로" : "최고 관리자로"}
                      </Button>
                      <Button
                        size="sm"
                        variant="danger"
                        busy={busy}
                        onClick={() => void revoke(admin)}
                      >
                        권한 해제
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      <p className="text-xs text-[var(--text-muted)]">
        권한 해제는 계정을 삭제하지 않습니다. 회원 계정은 그대로 유지되며 운영 권한만
        사라집니다.
      </p>
    </div>
  );
}
