"use client";

/**
 * Building a campaign link that the dashboard will actually recognise.
 *
 * The point of this utility is not saving typing — it is that a link
 * typed by hand is the single most common way attribution goes wrong.
 * `utm_source=Instagram` and `utm_source=instagram ` become two rows in
 * a report; a missing `utm_medium` makes an ad indistinguishable from a
 * profile link. So values are normalised here the same way the server
 * normalises them, and the preview shows exactly what will be counted.
 *
 * The destination stays on BOORDA. A UTM builder that can point
 * anywhere is a link generator for somebody else's site, and ours would
 * be the name on it.
 */

import { useMemo, useState } from "react";

import { Button, Card, cx, inputClass, labelClass } from "@/components/ui";
import { useToast } from "@/components/ui/Toast";

const ORIGIN = "https://boorda.kr";

/**
 * Ready-made parameter sets for the links BOORDA actually builds.
 *
 * Labels are conveniences; every preset produces standard UTM
 * parameters and nothing bespoke, so a link from here is readable by
 * any analytics tool BOORDA might add later.
 */
const PRESETS: { label: string; source: string; medium: string }[] = [
  { label: "Instagram 광고", source: "instagram", medium: "paid_social" },
  { label: "Instagram 프로필", source: "instagram", medium: "social" },
  { label: "YouTube 광고", source: "youtube", medium: "paid_video" },
  { label: "YouTube 설명란", source: "youtube", medium: "referral" },
  { label: "Google 광고", source: "google", medium: "cpc" },
  { label: "네이버", source: "naver", medium: "referral" },
  { label: "이메일", source: "boorda", medium: "email" },
  { label: "제휴", source: "partner", medium: "affiliate" },
];

/** The same normalisation the server applies, so the preview is honest. */
export function normaliseValue(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .slice(0, 120);
}

/** A path on BOORDA, whatever was typed. */
export function normaliseDestination(path: string): string {
  const trimmed = path.trim();
  if (!trimmed || trimmed === "/") return "/";
  // Anything that looks like another origin is reduced to its path:
  // this builder cannot be pointed off BOORDA.
  const withoutOrigin = trimmed.replace(/^https?:\/\/[^/]+/i, "");
  const rooted = withoutOrigin.startsWith("/") ? withoutOrigin : `/${withoutOrigin}`;
  return rooted.split("?")[0].split("#")[0];
}

export function buildCampaignUrl(input: {
  destination: string;
  source: string;
  medium: string;
  campaign: string;
  content?: string;
  term?: string;
}): string {
  const params = new URLSearchParams();
  const add = (key: string, value: string | undefined) => {
    const cleaned = normaliseValue(value ?? "");
    if (cleaned) params.set(key, cleaned);
  };
  add("utm_source", input.source);
  add("utm_medium", input.medium);
  add("utm_campaign", input.campaign);
  add("utm_content", input.content);
  add("utm_term", input.term);

  const query = params.toString();
  return `${ORIGIN}${normaliseDestination(input.destination)}${query ? `?${query}` : ""}`;
}

export function CampaignLinkBuilder() {
  const [destination, setDestination] = useState("/");
  const [source, setSource] = useState("instagram");
  const [medium, setMedium] = useState("paid_social");
  const [campaign, setCampaign] = useState("");
  const [content, setContent] = useState("");
  const [term, setTerm] = useState("");
  const { notify, notifyError } = useToast();

  const url = useMemo(
    () => buildCampaignUrl({ destination, source, medium, campaign, content, term }),
    [destination, source, medium, campaign, content, term],
  );
  const ready = Boolean(normaliseValue(source) && normaliseValue(medium) && normaliseValue(campaign));

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url);
      notify("링크를 복사했습니다.");
    } catch {
      notifyError("복사에 실패했습니다. 링크를 직접 선택해 주세요.");
    }
  };

  return (
    <Card className="flex flex-col gap-4 p-5">
      <div className="flex flex-col gap-1">
        <h2 className="text-sm font-semibold text-[var(--text-primary)]">캠페인 링크 만들기</h2>
        <p className="text-xs text-[var(--text-muted)]">
          값은 소문자로 정리되어 저장됩니다. 같은 캠페인이 여러 행으로 나뉘지 않도록 하기
          위해서입니다.
        </p>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((preset) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => {
              setSource(preset.source);
              setMedium(preset.medium);
            }}
            className={cx(
              "rounded-[var(--radius-md)] px-2.5 py-1 text-xs transition-colors",
              source === preset.source && medium === preset.medium
                ? "bg-[var(--accent-muted)] text-[var(--accent)]"
                : "text-[var(--text-secondary)] hover:bg-[var(--surface-sunken)]",
            )}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Field id="cl-destination" label="연결 경로" value={destination} onChange={setDestination} />
        <Field id="cl-source" label="소스 (utm_source)" value={source} onChange={setSource} />
        <Field id="cl-medium" label="매체 (utm_medium)" value={medium} onChange={setMedium} />
        <Field id="cl-campaign" label="캠페인 (utm_campaign)" value={campaign} onChange={setCampaign} />
        <Field id="cl-content" label="콘텐츠 (선택)" value={content} onChange={setContent} />
        <Field id="cl-term" label="키워드 (선택)" value={term} onChange={setTerm} />
      </div>

      <div className="flex flex-col gap-2">
        <label className={labelClass} htmlFor="cl-result">
          생성된 링크
        </label>
        <div className="flex flex-wrap items-center gap-2">
          <input
            id="cl-result"
            readOnly
            className={cx(inputClass, "min-w-0 flex-1 font-mono text-xs")}
            value={url}
          />
          <Button type="button" variant="primary" onClick={() => void copy()} disabled={!ready}>
            복사
          </Button>
        </div>
        {!ready ? (
          <p className="text-xs text-[var(--text-muted)]">
            소스 · 매체 · 캠페인을 모두 입력하면 복사할 수 있습니다.
          </p>
        ) : null}
      </div>
    </Card>
  );
}

function Field({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (next: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className={labelClass} htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        className={inputClass}
        value={value}
        maxLength={120}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  );
}
