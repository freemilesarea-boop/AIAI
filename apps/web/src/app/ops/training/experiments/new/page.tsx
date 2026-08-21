"use client";

/**
 * Create an experiment. Nothing is trained and nothing is queued.
 *
 * The form offers exactly the fields Phase 25's `Experiment` records —
 * Step 8's "do not add fields the backend ignores". A control that
 * looked like a setting and did nothing is worse than a missing one: it
 * appears in the record, invites an operator to reason about it, and has
 * no effect on anything.
 *
 * The base model is chosen from what is registered rather than typed. An
 * experiment on a model nobody registered cannot be created, and finding
 * that out from a 409 after writing a hypothesis is a worse experience
 * than not being offered it.
 */

import { useRouter } from "next/navigation";
import { useState } from "react";

import { OpsHeader } from "@/components/ops/OpsShell";
import { OpsEmpty, Panel, PanelError, SectionSkeleton } from "@/components/ops/primitives";
import { useOpsResource } from "@/components/ops/useOpsResource";
import { Button, errorClass, hintClass, inputClass, labelClass } from "@/components/ui";
import { OpsError, ops } from "@/lib/ops/client";

export default function NewExperimentPage() {
  const router = useRouter();
  const catalogue = useOpsResource(() => ops.catalogue());

  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [description, setDescription] = useState("");
  const [baseModel, setBaseModel] = useState("");
  const [operator, setOperator] = useState("");
  const [tags, setTags] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const models = catalogue.data?.base_models ?? [];
  const selected = baseModel || models[0]?.model_id || "";
  const ready = name.trim().length > 0 && hypothesis.trim().length > 0 && selected.length > 0;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await ops.createExperiment({
        name: name.trim(),
        hypothesis: hypothesis.trim(),
        base_model_id: selected,
        description: description.trim(),
        operator: operator.trim(),
        tags: tags
          .split(",")
          .map((tag) => tag.trim())
          .filter(Boolean),
      });
      router.push(`/ops/training/experiments/${created.experiment_id}`);
    } catch (caught) {
      setError(
        caught instanceof OpsError ? caught.message : "The experiment could not be created.",
      );
      setBusy(false);
    }
  };

  return (
    <>
      <OpsHeader
        title="New experiment"
        breadcrumb={[{ href: "/ops/training/experiments", label: "Experiments" }]}
        description="An experiment records a hypothesis. Creating one starts no training and queues nothing."
      />

      <Panel title="Hypothesis" id="create">
        {catalogue.loading && <SectionSkeleton rows={4} />}
        {catalogue.error && <PanelError message={catalogue.error} onRetry={catalogue.refresh} />}

        {catalogue.data && models.length === 0 ? (
          <OpsEmpty
            title="No model baseline is registered"
            description="An experiment is built on a model. Register one with `luber-training baseline register` before creating an experiment."
          />
        ) : (
          catalogue.data && (
            <form onSubmit={submit} className="max-w-2xl space-y-5">
              <div>
                <label htmlFor="experiment-name" className={labelClass}>
                  Name
                </label>
                <input
                  id="experiment-name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  maxLength={200}
                  required
                  className={inputClass}
                />
                <p className={hintClass}>How this experiment will be referred to in a report.</p>
              </div>

              <div>
                <label htmlFor="experiment-hypothesis" className={labelClass}>
                  Hypothesis
                </label>
                <textarea
                  id="experiment-hypothesis"
                  value={hypothesis}
                  onChange={(event) => setHypothesis(event.target.value)}
                  rows={3}
                  maxLength={2000}
                  required
                  className={inputClass}
                />
                <p className={hintClass}>
                  What you expect to be true, and about what. Qualification later checks that a
                  candidate addressed its own claim, so a vague hypothesis is one nothing can
                  answer.
                </p>
              </div>

              <div>
                <label htmlFor="experiment-model" className={labelClass}>
                  Base model
                </label>
                <select
                  id="experiment-model"
                  value={selected}
                  onChange={(event) => setBaseModel(event.target.value)}
                  className={inputClass}
                >
                  {models.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.model_name} {model.model_version} · {model.stage} ·{" "}
                      {model.model_id}
                    </option>
                  ))}
                </select>
                <p className={hintClass}>
                  Only registered baselines are offered. A model is identified by its upstream
                  commit and declared variant.
                </p>
              </div>

              <div>
                <label htmlFor="experiment-description" className={labelClass}>
                  Description
                </label>
                <textarea
                  id="experiment-description"
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  rows={3}
                  maxLength={8000}
                  className={inputClass}
                />
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="experiment-operator" className={labelClass}>
                    Operator
                  </label>
                  <input
                    id="experiment-operator"
                    value={operator}
                    onChange={(event) => setOperator(event.target.value)}
                    maxLength={120}
                    className={inputClass}
                  />
                </div>
                <div>
                  <label htmlFor="experiment-tags" className={labelClass}>
                    Tags
                  </label>
                  <input
                    id="experiment-tags"
                    value={tags}
                    onChange={(event) => setTags(event.target.value)}
                    placeholder="korean, vocal"
                    className={inputClass}
                  />
                  <p className={hintClass}>Comma separated.</p>
                </div>
              </div>

              {error && (
                <p role="alert" className={errorClass}>
                  {error}
                </p>
              )}

              <div className="flex gap-2">
                <Button type="submit" variant="primary" disabled={!ready} busy={busy}>
                  Create experiment
                </Button>
                <Button type="button" variant="ghost" onClick={() => router.back()}>
                  Cancel
                </Button>
              </div>
            </form>
          )
        )}
      </Panel>
    </>
  );
}
