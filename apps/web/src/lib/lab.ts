/**
 * BOORDA LAB — the shape of a model catalogue, with nothing in it yet.
 *
 * LAB is where a model lives before it becomes the default: an
 * experiment, then a beta, then stable, then the thing everyone gets.
 * That lifecycle is real and worth modelling now; the models are not.
 * Nothing in BOORDA can currently switch generation models, so every
 * entry below is marked unavailable and no card offers a working action.
 *
 * When a model-catalogue API arrives it returns a `LabCatalog` — the
 * same shape this file exports as a constant — and the page renders it
 * without changing. `PREVIEW_ENTRIES` is then deleted, not edited.
 *
 * The rule this file exists to enforce: an entry that is not wired to a
 * backend must be visibly unavailable. A card that looks usable and is
 * not costs more trust than an empty page.
 */

/**
 * Where a model sits in its life.
 *
 * `STABLE` means shipped and dependable. `DEFAULT` is not a status —
 * it is a property of the catalogue (exactly one model is default), so
 * it lives on `LabCatalog` rather than here.
 */
export type ModelStatus = "NEW" | "BETA" | "EXPERIMENTAL" | "COMING_SOON" | "STABLE";

/** Progression order. Used for sorting and for the lifecycle legend. */
export const LIFECYCLE: readonly ModelStatus[] = [
  "EXPERIMENTAL",
  "BETA",
  "NEW",
  "STABLE",
  "COMING_SOON",
];

export interface StatusPresentation {
  label: string;
  /** Tailwind classes for the badge. Reads from design tokens only. */
  className: string;
}

export const STATUS_PRESENTATION: Record<ModelStatus, StatusPresentation> = {
  NEW: {
    label: "NEW",
    className: "bg-[var(--brand-muted)] text-[var(--brand-text)]",
  },
  BETA: {
    label: "BETA",
    className: "bg-[var(--surface-overlay)] text-[var(--text-primary)]",
  },
  EXPERIMENTAL: {
    label: "EXPERIMENTAL",
    className: "bg-[var(--danger-muted)] text-[var(--danger)]",
  },
  COMING_SOON: {
    label: "COMING SOON",
    className: "bg-[var(--surface-sunken)] text-[var(--text-muted)]",
  },
  STABLE: {
    label: "STABLE",
    className: "bg-[var(--surface-overlay)] text-[var(--text-secondary)]",
  },
};

export interface LabEntry {
  id: string;
  /** Product-facing name. Not a checkpoint id. */
  name: string;
  /** Version string as users should see it, e.g. "v1". */
  version: string;
  status: ModelStatus;
  /** One or two sentences: what this is. */
  description: string;
  /** What it improves over the current default. Short bullets. */
  improvements: string[];
  /** ISO date, or `null` when unreleased. */
  releaseDate: string | null;
  /**
   * Whether a user can actually use this right now.
   *
   * `false` for every entry today: BOORDA has no model-switching API,
   * so nothing here is selectable. The card renders its CTA disabled
   * and says why.
   */
  available: boolean;
  /** Engineering detail for people who want it. Optional. */
  technicalNotes?: string;
  /**
   * Shown when the model may behave unpredictably. Present on anything
   * EXPERIMENTAL; the card renders it as a distinct warning, not as
   * body copy.
   */
  experimentalWarning?: string;
}

export interface LabCatalog {
  entries: readonly LabEntry[];
  /**
   * The model used when the user picks nothing. `null` while there is
   * no catalogue API — the product generates with whatever the backend
   * is configured for, and the UI must not claim to know which.
   */
  defaultModelId: string | null;
  /**
   * Ids the signed-in user may opt into. Empty until model switching
   * exists. Kept separate from `available` so the future API can gate
   * a beta per-account without rewriting the entry.
   */
  accessibleModelIds: readonly string[];
}

/**
 * Example entries, marked as examples.
 *
 * These describe work that is either shipped in the backend but not
 * selectable (BOORDA Music) or genuinely not delivered yet (the two
 * experiments). None is usable, and each says so on its face.
 */
export const PREVIEW_ENTRIES: readonly LabEntry[] = [
  {
    id: "boorda-music",
    name: "BOORDA Music",
    version: "v1",
    status: "STABLE",
    description: "현재 음악 생성에 사용되는 기본 모델입니다.",
    improvements: ["가사와 분위기 설명을 함께 반영", "완성된 마스터 트랙 출력"],
    releaseDate: null,
    available: false,
    technicalNotes: "모델 선택 기능이 아직 없어 LAB에서는 전환할 수 없습니다.",
  },
  {
    id: "high-frequency-detail",
    name: "High-Frequency Detail",
    version: "v0",
    status: "EXPERIMENTAL",
    description: "고음역의 질감과 공기감을 개선하기 위한 실험용 모델입니다.",
    improvements: ["6–16 kHz 대역의 자연스러운 질감", "금속적인 울림 감소"],
    releaseDate: null,
    available: false,
    technicalNotes: "연구 단계입니다. 아직 검증된 개선 결과가 없습니다.",
    experimentalWarning:
      "실험용입니다. 결과가 불안정할 수 있고, 예고 없이 변경되거나 중단될 수 있습니다.",
  },
  {
    id: "vocal-detail",
    name: "Vocal Detail",
    version: "v0",
    status: "EXPERIMENTAL",
    description: "보컬의 표현력과 디테일을 개선하기 위한 실험용 모델입니다.",
    improvements: ["보컬 질감 표현", "자연스러운 발음과 호흡"],
    releaseDate: null,
    available: false,
    technicalNotes: "연구 단계입니다. 아직 검증된 개선 결과가 없습니다.",
    experimentalWarning:
      "실험용입니다. 결과가 불안정할 수 있고, 예고 없이 변경되거나 중단될 수 있습니다.",
  },
];

/**
 * The catalogue as the product currently knows it.
 *
 * Everything is unavailable and there is no default, because no API
 * reports either. Replace the body with a fetch when one exists; the
 * return type does not change.
 */
export function labCatalog(): LabCatalog {
  return {
    entries: PREVIEW_ENTRIES,
    defaultModelId: null,
    accessibleModelIds: [],
  };
}

/**
 * Whether the user can act on an entry.
 *
 * Both conditions must hold: the entry is generally available *and*
 * this account may reach it. Written as one function so no card can
 * accidentally check only half of it.
 */
export function isUsable(entry: LabEntry, catalog: LabCatalog): boolean {
  return entry.available && catalog.accessibleModelIds.includes(entry.id);
}

export function formatReleaseDate(iso: string | null): string {
  if (iso === null) return "미정";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "미정";
  return new Intl.DateTimeFormat("ko-KR", { dateStyle: "long" }).format(date);
}
