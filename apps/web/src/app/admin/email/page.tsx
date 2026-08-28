"use client";

/**
 * Composing an announcement, and being told plainly that it did not go
 * out.
 *
 * BOORDA has no email provider configured — not in the repository, not
 * in the deployment. So this page composes a campaign, resolves its
 * audience to a number on the server, and saves a draft. It does not
 * offer a send button.
 *
 * That is the honest version. The dishonest version is a send button
 * that stores a row and shows a success toast, and its cost lands on the
 * operator who believes they announced a price change and did not.
 *
 * The recipient count comes from the server, not from counting a page of
 * results in the browser: it is the number an operator would confirm
 * against before a send, so it has to be a server-side fact.
 */

import { useCallback, useEffect, useState } from "react";

import { Button, Card, Skeleton, inputClass, labelClass } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";
import {
  PLAN_LABELS,
  createCampaign,
  fetchCampaigns,
  formatCount,
  formatDateTime,
  previewAudience,
  type AudienceType,
  type Campaign,
  type CampaignDraft,
} from "@/lib/admin";

const AUDIENCES: { value: AudienceType; label: string }[] = [
  { value: "ALL", label: "전체 회원" },
  { value: "PLAN", label: "특정 요금제" },
];

const PLANS = ["free", "basic", "pro", "creator"];

export default function AdminEmailPage() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [audience, setAudience] = useState<AudienceType>("ALL");
  const [planId, setPlanId] = useState("basic");
  const [recipients, setRecipients] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [campaigns, setCampaigns] = useState<Campaign[] | null>(null);
  const { notify, notifyError } = useToast();

  const draft: CampaignDraft = {
    subject: subject.trim() || "(제목 없음)",
    body: body.trim() || "(내용 없음)",
    audience_type: audience,
    plan_id: audience === "PLAN" ? planId : null,
  };

  const reload = useCallback(async (signal?: AbortSignal) => {
    try {
      // Defaulted rather than trusted: a response missing `items` should
      // render an empty list, not throw inside render and take the
      // compose form down with it.
      setCampaigns((await fetchCampaigns(signal)).items ?? []);
    } catch {
      if (signal?.aborted) return;
      setCampaigns([]);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void reload(controller.signal);
    return () => controller.abort();
  }, [reload]);

  // Recomputed whenever the audience changes, never derived in the
  // browser from a page of users.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const { recipient_count } = await previewAudience({
          audience_type: audience,
          plan_id: audience === "PLAN" ? planId : null,
        });
        if (!cancelled) setRecipients(recipient_count);
      } catch {
        if (!cancelled) setRecipients(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [audience, planId]);

  const save = async () => {
    setBusy(true);
    try {
      await createCampaign(draft);
      await reload();
      notify("초안을 저장했습니다. 발송은 되지 않았습니다.");
      setSubject("");
      setBody("");
    } catch {
      notifyError("초안 저장에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--text-primary)]">
          이메일 발송
        </h1>
        <p className="text-sm text-[var(--text-secondary)]">
          공지 초안을 작성하고 대상 인원을 확인합니다.
        </p>
      </header>

      <div
        role="note"
        className="rounded-[var(--radius-md)] border border-[var(--border-default)] bg-[var(--surface-sunken)] p-4 text-sm text-[var(--text-secondary)]"
      >
        <strong className="font-medium text-[var(--text-primary)]">
          현재 이메일 발송은 제공되지 않습니다.
        </strong>{" "}
        메일 발송 서비스가 연결되어 있지 않아 작성한 내용은 초안으로만 저장됩니다. 실제 발송은
        이루어지지 않습니다.
      </div>

      <Card className="flex flex-col gap-4 p-5">
        <form
          className="flex flex-col gap-4"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <div className="flex flex-col gap-1.5">
            <label className={labelClass} htmlFor="campaign-subject">
              제목
            </label>
            <input
              id="campaign-subject"
              className={inputClass}
              value={subject}
              maxLength={200}
              onChange={(event) => setSubject(event.target.value)}
              required
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className={labelClass} htmlFor="campaign-body">
              내용
            </label>
            <textarea
              id="campaign-body"
              className={inputClass}
              rows={8}
              value={body}
              onChange={(event) => setBody(event.target.value)}
              required
            />
            <p className="text-xs text-[var(--text-muted)]">
              일반 텍스트로 저장됩니다. HTML은 지원하지 않습니다.
            </p>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1.5">
              <label className={labelClass} htmlFor="campaign-audience">
                대상
              </label>
              <select
                id="campaign-audience"
                className={inputClass}
                value={audience}
                onChange={(event) => setAudience(event.target.value as AudienceType)}
              >
                {AUDIENCES.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>

            {audience === "PLAN" ? (
              <div className="flex flex-col gap-1.5">
                <label className={labelClass} htmlFor="campaign-plan">
                  요금제
                </label>
                <select
                  id="campaign-plan"
                  className={inputClass}
                  value={planId}
                  onChange={(event) => setPlanId(event.target.value)}
                >
                  {PLANS.map((plan) => (
                    <option key={plan} value={plan}>
                      {PLAN_LABELS[plan] ?? plan}
                    </option>
                  ))}
                </select>
              </div>
            ) : null}

            <p className="text-sm text-[var(--text-secondary)]" aria-live="polite">
              대상 인원:{" "}
              <strong className="tabular-nums text-[var(--text-primary)]">
                {recipients === null ? "확인 중…" : `${formatCount(recipients)}명`}
              </strong>
            </p>
          </div>

          <div>
            <Button type="submit" variant="primary" busy={busy}>
              초안 저장
            </Button>
          </div>
        </form>
      </Card>

      <section className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">작성한 초안</h2>
        {campaigns === null ? (
          <Skeleton className="h-24 w-full" />
        ) : campaigns.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">아직 작성한 초안이 없습니다.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {campaigns.map((campaign) => (
              <li key={campaign.id}>
                <Card className="flex flex-col gap-1 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium text-[var(--text-primary)]">
                      {campaign.subject}
                    </span>
                    <span className="rounded-[var(--radius-full)] bg-[var(--surface-sunken)] px-2 py-0.5 text-[11px] text-[var(--text-muted)]">
                      {campaign.status === "DRAFT" ? "초안 (미발송)" : campaign.status}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-muted)]">
                    대상 {formatCount(campaign.recipient_count)}명 ·{" "}
                    {formatDateTime(campaign.created_at)} · {campaign.created_by_email}
                  </p>
                </Card>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
