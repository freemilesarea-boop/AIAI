# Dataset & Third-Party Service Policy

## Absolute prohibitions

The following are forbidden anywhere in this codebase or operations:

1. Calling the Suno API unofficially.
2. Reverse engineering Suno's internal web APIs.
3. Forcibly extracting Suno audio URLs.
4. Auto-scraping Suno public songs as a dataset.
5. Auto-collecting Suno output to train competing models.
6. Stealing or bypassing any service's auth tokens, cookies, or API keys.

Suno's terms restrict using its output in competing products/services,
so Suno data is **not** a training dataset for this project. Suno-related
data may only be considered for ingestion if **all** of the following
hold: required content rights secured, ML training contractually
allowed, required platform permission secured, and provenance
documentation exists.

## Training data eligibility

Priority of sources:

- **A** — Audio we produce with all rights secured.
- **B** — Audio secured via explicit AI-training contracts with
  composers/producers/vocalists.
- **C** — Datasets explicitly licensed for commercial ML training.
- **D** — Catalogs licensed via separate agreements.
- **E** — Material in a directory the operator explicitly authorised for
  training. The weakest of the five and recorded as such: basis
  `OPERATOR_AUTHORIZED_SCOPE`, rights status `OPERATOR_AUTHORIZED`, and
  `lyrics_rights_confirmed` / `performer_rights_confirmed` left false
  because nobody produced a publisher clearance or a performer
  agreement. It authorises the named scope and nothing beyond it. See
  `OPERATOR_AUTHORIZED_TRAINING_DATA.md`.

Hard rules:

- A file with no license metadata **fails** dataset validation. An
  operator authorisation is metadata only when it is actually recorded:
  a source, a scope and a date. A folder name is still not a licence.
- `commercial_training_allowed = false` ⇒ automatically excluded from
  production training pipelines.
- Every dataset asset records provenance: source, creator, owner,
  license, SHA256 (schema in `DATASET_PIPELINE.md`).

## Product safety

- No voice-clone features for specific real artists (e.g. "IU's voice",
  "Bruno Mars voice"). UI offers only generic vocal descriptors
  (female/male, soft, powerful, breathy, warm).
- Artist-name prompts are normalized to musical characteristics via the
  prompt normalization layer, not identity reproduction.
