"use client";

/**
 * Create a run. Four separate steps, and this is only the first.
 *
 * Step 55: Create, Validate, Stage and Dispatch stay apart. One "TRAIN"
 * button that did all four would make the expensive, irreversible part
 * — sending approved data to a rented machine and starting a trainer —
 * indistinguishable from writing a record. Creating a run here writes a
 * DRAFT and nothing else; validation and dispatch happen on the run's
 * own page, deliberately.
 *
 * Nothing is typed that could be a path. The dataset and curation are
 * chosen from what the deployment configured, by identifier, and every
 * digest on the resulting run is read from the lock rather than stated
 * by the browser.
 *
 * Incompatible workers are shown, disabled, with the reason a probe
 * established. Hiding them would leave an operator wondering where the
 * machine they rented went.
 */

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import {
  KeyValue,
  OpsEmpty,
  OpsStatus,
  Panel,
  PanelError,
  SectionSkeleton,
} from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { Button, errorClass, hintClass, inputClass, labelClass } from "@/components/ui";
import { OpsError, ops } from "@/lib/ops/client";
import { decimal, num, shortDigest } from "@/lib/ops/format";

function NewRunForm() {
  const router = useRouter();
  const search = useSearchParams();

  const catalogue = useOpsResource(() => ops.catalogue());
  const experiments = useOpsResource(() => ops.experiments({ limit: 200 }));

  const [experimentId, setExperimentId] = useState(search.get("experiment") ?? "");
  const [datasetId, setDatasetId] = useState("");
  const [curationId, setCurationId] = useState("");
  const [preset, setPreset] = useState("LORA_STANDARD");
  const [backend, setBackend] = useState("dry-run");
  const [workerId, setWorkerId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const compatibility = useOpsResource(() => ops.workerCompatibility(backend), {
    deps: [backend],
  });

  // Default to the single available option rather than making an
  // operator choose from a list of one.
  useEffect(() => {
    if (!datasetId && catalogue.data?.datasets.length) {
      setDatasetId(catalogue.data.datasets[0].build_id);
    }
    if (!curationId && catalogue.data?.curations.length) {
      setCurationId(catalogue.data.curations[0].build_id);
    }
  }, [catalogue.data, datasetId, curationId]);

  const dataset = catalogue.data?.datasets.find((item) => item.build_id === datasetId);
  const curation = catalogue.data?.curations.find((item) => item.build_id === curationId);

  // Caught before creation rather than left to the gate: a run that
  // failed because two selections did not belong together is noise in
  // an experiment's history.
  const mismatch =
    dataset && curation && curation.source_dataset_lock_sha256 && dataset.lock_sha256
      ? curation.source_dataset_lock_sha256 !== dataset.lock_sha256
      : false;

  const ready =
    experimentId.length > 0 && datasetId.length > 0 && curationId.length > 0 && !mismatch;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await ops.createRun({
        experiment_id: experimentId,
        dataset_build_id: datasetId,
        curation_build_id: curationId,
        preset,
        execution_backend: backend,
        worker_id: workerId || null,
      });
      router.push(`/ops/training/runs/${created.run.run_id}`);
    } catch (caught) {
      setError(caught instanceof OpsError ? caught.message : "The run could not be created.");
      setBusy(false);
    }
  };

  const loading = catalogue.loading || experiments.loading;
  const noBuilds =
    catalogue.data && catalogue.data.datasets.length === 0 && catalogue.data.curations.length === 0;

  return (
    <>
      <OpsHeader
        title="Create a training run"
        breadcrumb={[{ href: "/ops/training/runs", label: "Runs" }]}
        description="This writes a DRAFT run. Validation, staging and dispatch are separate, deliberate steps on the run's own page."
      />

      <div className="grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Panel title="Run" id="create">
          {loading && <SectionSkeleton rows={6} />}
          {catalogue.error && <PanelError message={catalogue.error} onRetry={catalogue.refresh} />}

          {noBuilds ? (
            <OpsEmpty
              title="No dataset or curation builds are available"
              description={
                (catalogue.data?.dataset_problems ?? [])
                  .concat(catalogue.data?.curation_problems ?? [])
                  .join(" ") ||
                "This deployment has no build roots configured, so there is nothing a run could be built from. Set OPS_DATASET_BUILDS_ROOT and OPS_CURATION_BUILDS_ROOT."
              }
            />
          ) : (
            catalogue.data && (
              <form onSubmit={submit} className="space-y-5">
                <div>
                  <label htmlFor="run-experiment" className={labelClass}>
                    Experiment
                  </label>
                  <select
                    id="run-experiment"
                    value={experimentId}
                    onChange={(event) => setExperimentId(event.target.value)}
                    required
                    className={inputClass}
                  >
                    <option value="">Choose an experiment…</option>
                    {(experiments.data?.items ?? []).map((item) => (
                      <option key={item.experiment_id} value={item.experiment_id}>
                        {item.name} · {item.status}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <label htmlFor="run-dataset" className={labelClass}>
                      Dataset build
                    </label>
                    <select
                      id="run-dataset"
                      value={datasetId}
                      onChange={(event) => setDatasetId(event.target.value)}
                      className={inputClass}
                    >
                      {catalogue.data.datasets.map((item) => (
                        <option key={item.build_id} value={item.build_id}>
                          {item.build_id} · {item.identity}
                        </option>
                      ))}
                    </select>
                    <p className={hintClass}>
                      Offered by identifier. No field on this page accepts a path.
                    </p>
                  </div>

                  <div>
                    <label htmlFor="run-curation" className={labelClass}>
                      Curation build
                    </label>
                    <select
                      id="run-curation"
                      value={curationId}
                      onChange={(event) => setCurationId(event.target.value)}
                      className={inputClass}
                    >
                      {catalogue.data.curations.map((item) => (
                        <option key={item.build_id} value={item.build_id}>
                          {item.build_id} · {item.identity}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {mismatch && (
                  <p role="alert" className={errorClass}>
                    This curation was built from a different dataset lock than the one selected.
                    The gates would refuse it; choose the pair that belong together.
                  </p>
                )}

                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <label htmlFor="run-preset" className={labelClass}>
                      Configuration preset
                    </label>
                    <select
                      id="run-preset"
                      value={preset}
                      onChange={(event) => setPreset(event.target.value)}
                      className={inputClass}
                    >
                      {catalogue.data.presets.map((item) => (
                        <option key={item.name} value={item.name}>
                          {item.name}
                        </option>
                      ))}
                    </select>
                    <p className={hintClass}>
                      {catalogue.data.presets.find((item) => item.name === preset)?.intent}
                    </p>
                  </div>

                  <div>
                    <label htmlFor="run-backend" className={labelClass}>
                      Execution backend
                    </label>
                    <select
                      id="run-backend"
                      value={backend}
                      onChange={(event) => {
                        setBackend(event.target.value);
                        setWorkerId("");
                      }}
                      className={inputClass}
                    >
                      {catalogue.data.backends.map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                    <p className={hintClass}>
                      The dry-run backend trains nothing and produces a MOCK checkpoint that can
                      never be evaluated.
                    </p>
                  </div>
                </div>

                <fieldset>
                  <legend className={labelClass}>Worker</legend>
                  {compatibility.loading && <SectionSkeleton rows={2} />}
                  {compatibility.data && compatibility.data.length === 0 && (
                    <p className={hintClass}>No workers are registered.</p>
                  )}
                  <div className="mt-2 space-y-2">
                    {(compatibility.data ?? []).map((row) => (
                      <label
                        key={row.worker.worker_id}
                        className="flex items-start gap-3 rounded-[var(--radius-md)] border border-[var(--border-subtle)] px-3 py-2"
                      >
                        <input
                          type="radio"
                          name="worker"
                          value={row.worker.worker_id}
                          checked={workerId === row.worker.worker_id}
                          disabled={!row.compatible}
                          onChange={(event) => setWorkerId(event.target.value)}
                          className="mt-1"
                        />
                        <span className="min-w-0 flex-1">
                          <span className="flex flex-wrap items-center gap-2">
                            <span className="text-sm text-[var(--text-primary)]">
                              {row.worker.name}
                            </span>
                            <OpsStatus status={row.worker.worker_class} />
                            <OpsStatus status={row.worker.liveness} />
                          </span>
                          {row.reasons.length > 0 && (
                            <span className="mt-1 block text-[11px] leading-relaxed text-[var(--text-muted)]">
                              {row.reasons.join("; ")}
                            </span>
                          )}
                        </span>
                      </label>
                    ))}
                  </div>
                </fieldset>

                {error && (
                  <p role="alert" className={errorClass}>
                    {error}
                  </p>
                )}

                <div className="flex gap-2">
                  <Button type="submit" variant="primary" disabled={!ready} busy={busy}>
                    Create run
                  </Button>
                  <Button type="button" variant="ghost" onClick={() => router.back()}>
                    Cancel
                  </Button>
                </div>
              </form>
            )
          )}
        </Panel>

        <Panel title="What was selected" subtitle="Digests read from the locks themselves.">
          {dataset || curation ? (
            <KeyValue
              columns={1}
              items={[
                { label: "Dataset", value: dataset?.identity ?? "—" },
                { label: "Dataset lock", value: shortDigest(dataset?.lock_sha256, 16) },
                { label: "Tracks", value: num(dataset?.track_count ?? null) },
                { label: "Curation", value: curation?.identity ?? "—" },
                {
                  label: "Curated manifest",
                  value: shortDigest(curation?.manifest_sha256, 16),
                },
                {
                  label: "Selected",
                  value: `${num(curation?.track_count ?? null)} track(s) · ${decimal(
                    curation?.hours ?? null,
                    2,
                  )} hours`,
                },
              ]}
            />
          ) : (
            <p className="text-xs text-[var(--text-muted)]">Choose a dataset and a curation.</p>
          )}
          <p className="mt-4 text-[11px] leading-relaxed text-[var(--text-muted)]">
            Creating a run transfers nothing and starts nothing. Rights and evaluation-leakage
            gates run at validation, against the files as they are then — not as they were when
            this page was loaded.
          </p>
        </Panel>
      </div>
    </>
  );
}

export default function NewRunPage() {
  // `useSearchParams` needs a Suspense boundary in the app router.
  return (
    <Suspense fallback={<SectionSkeleton rows={6} />}>
      <NewRunForm />
    </Suspense>
  );
}
