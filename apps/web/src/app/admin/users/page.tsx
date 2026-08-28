"use client";

/**
 * Members, and what one account has done.
 *
 * The console shows counts, never content: an operator can see that an
 * account generated forty tracks and downloaded twelve, and cannot open
 * any of them. Support work needs the shape of someone's usage, not
 * their music.
 *
 * There is no delete control, and no route behind one. Closing an
 * account is the customer's own action in Settings, and it anonymises
 * rather than erases — a console button that removed a person on an
 * operator's behalf would be the most dangerous control in the product
 * sitting behind the least specific intent.
 */

import { useEffect, useMemo, useState } from "react";

import { Button, Card, EmptyState, Skeleton, inputClass } from "@/components/ui";
import {
  ROLE_LABELS,
  fetchUser,
  fetchUsers,
  formatCount,
  formatDateTime,
  planLabel,
  type AdminUser,
  type AdminUserDetail,
} from "@/lib/admin";

const PAGE_SIZE = 25;

function RolePill({ role }: { role: AdminUser["role"] }) {
  if (role === "USER") return null;
  return (
    <span className="rounded-[var(--radius-full)] bg-[var(--accent-muted)] px-2 py-0.5 text-[11px] font-medium text-[var(--accent)]">
      {ROLE_LABELS[role]}
    </span>
  );
}

function Detail({ id, onClose }: { id: string; onClose: () => void }) {
  const [detail, setDetail] = useState<AdminUserDetail | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    void (async () => {
      try {
        setDetail(await fetchUser(id, controller.signal));
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [id]);

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex items-start justify-between gap-3">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">회원 상세</h2>
        <Button variant="ghost" onClick={onClose}>
          닫기
        </Button>
      </div>
      {failed ? (
        <p className="text-sm text-[var(--danger)]">회원 정보를 불러오지 못했습니다.</p>
      ) : detail === null ? (
        <Skeleton className="h-24 w-full" />
      ) : (
        <>
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div>
              <dt className="text-xs text-[var(--text-muted)]">이메일</dt>
              <dd className="truncate text-[var(--text-primary)]">{detail.user.email}</dd>
            </div>
            <div>
              <dt className="text-xs text-[var(--text-muted)]">요금제</dt>
              <dd className="text-[var(--text-primary)]">
                {planLabel(detail.user.plan_id)}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-[var(--text-muted)]">가입일</dt>
              <dd className="text-[var(--text-primary)]">
                {formatDateTime(detail.user.created_at)}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-[var(--text-muted)]">권한</dt>
              <dd className="text-[var(--text-primary)]">{ROLE_LABELS[detail.user.role]}</dd>
            </div>
          </dl>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "생성 요청", value: detail.activity.generations },
              { label: "생성 완료", value: detail.activity.completed },
              { label: "다운로드", value: detail.activity.downloads },
              { label: "결제", value: detail.activity.payments },
            ].map((row) => (
              <div
                key={row.label}
                className="rounded-[var(--radius-md)] bg-[var(--surface-sunken)] p-3"
              >
                <p className="text-xs text-[var(--text-muted)]">{row.label}</p>
                <p className="text-lg font-semibold tabular-nums text-[var(--text-primary)]">
                  {formatCount(row.value)}
                </p>
              </div>
            ))}
          </div>
          <p className="text-xs text-[var(--text-muted)]">
            건수만 표시합니다. 회원이 만든 음악은 운영 콘솔에서 열람할 수 없습니다.
          </p>
        </>
      )}
    </Card>
  );
}

export default function AdminUsersPage() {
  const [search, setSearch] = useState("");
  const [applied, setApplied] = useState("");
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ items: AdminUser[]; total: number } | null>(null);
  const [failed, setFailed] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setFailed(false);
    void (async () => {
      try {
        setData(
          await fetchUsers(
            { search: applied || undefined, limit: PAGE_SIZE, offset: page * PAGE_SIZE },
            controller.signal,
          ),
        );
      } catch {
        if (controller.signal.aborted) return;
        setFailed(true);
      }
    })();
    return () => controller.abort();
  }, [applied, page]);

  const pages = useMemo(
    () => (data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1),
    [data],
  );

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">회원</h1>
        <p className="text-sm text-[var(--text-secondary)]">
          {data ? `${formatCount(data.total)}명` : "불러오는 중…"}
        </p>
      </header>

      <form
        className="flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          setPage(0);
          setApplied(search.trim());
        }}
      >
        <input
          className={inputClass}
          type="search"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="이메일 또는 회원 ID"
          aria-label="회원 검색"
        />
        <Button type="submit">검색</Button>
      </form>

      {selected ? <Detail id={selected} onClose={() => setSelected(null)} /> : null}

      {failed ? (
        <EmptyState title="회원 목록을 불러오지 못했습니다" description="잠시 후 다시 시도해 주세요." />
      ) : data === null ? (
        <Skeleton className="h-64 w-full" />
      ) : data.items.length === 0 ? (
        <EmptyState title="검색 결과가 없습니다" description="다른 조건으로 검색해 보세요." />
      ) : (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[36rem] text-sm">
            <caption className="sr-only">회원 목록</caption>
            <thead>
              <tr className="border-b border-[var(--border-default)] text-left text-xs text-[var(--text-muted)]">
                <th scope="col" className="p-3 font-medium">
                  이메일
                </th>
                <th scope="col" className="p-3 font-medium">
                  요금제
                </th>
                <th scope="col" className="p-3 font-medium">
                  가입일
                </th>
                <th scope="col" className="p-3 font-medium">
                  <span className="sr-only">상세</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((user) => (
                <tr key={user.id} className="border-b border-[var(--border-default)] last:border-0">
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-[var(--text-primary)]">{user.email}</span>
                      <RolePill role={user.role} />
                    </div>
                  </td>
                  <td className="p-3 text-[var(--text-secondary)]">
                    {planLabel(user.plan_id)}
                  </td>
                  <td className="p-3 text-[var(--text-secondary)]">
                    {formatDateTime(user.created_at)}
                  </td>
                  <td className="p-3 text-right">
                    <Button variant="ghost" onClick={() => setSelected(user.id)}>
                      상세
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {pages > 1 ? (
        <div className="flex items-center justify-between text-sm">
          <Button variant="ghost" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
            이전
          </Button>
          <span className="text-[var(--text-muted)]">
            {page + 1} / {pages}
          </span>
          <Button
            variant="ghost"
            disabled={page + 1 >= pages}
            onClick={() => setPage((p) => p + 1)}
          >
            다음
          </Button>
        </div>
      ) : null}
    </div>
  );
}
