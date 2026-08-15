# Generation lineage — audit

Read from the code, the migrations and the live database before any
Phase 17 code was written, because Step 1 gates the rest: no schema may
be added until this audit proves it necessary.

It does not. The taxonomy Phase 17 needs is already derivable from
durable fields. Two other findings matter more than that one, and both
change what the implementation has to do.

---

## 1. Durable provenance that exists today

`generations` carries, per row:

| Column | Written by |
|---|---|
| `parent_generation_id` | Generate Again (from the request), Extend, Replace Section, Cover |
| `generation_group_id` | every create; siblings of one CREATE share it |
| `edit_kind` | Extend (`EXTEND`), Replace Section (`REPLACE_RANGE`), Cover (`COVER`) |
| `edit_start_seconds` / `edit_end_seconds` | Replace Section, Extend |
| `variation_label` | optional free label, never load-bearing |
| `reference_audio_id` | reference-conditioned generation (Phase 15R) |
| `source_adherence` | Cover strength (Phase 13D-2) |

Write sites, all in `apps/api/src/luber_api/routes/generations.py`:
Cover at 698-700, Replace Section at 774-776, Extend at 838-843, and the
ordinary create at 345-347 which passes `parent_generation_id` straight
through from the request.

## 2. Operation taxonomy — derivable, no migration needed

Every conceptual operation is distinguishable from durable fields alone:

| Operation | `edit_kind` | `parent_generation_id` |
|---|---|---|
| `ORIGINAL` | NULL | NULL |
| `GENERATE_AGAIN` | NULL | set |
| `EXTEND` | `EXTEND` | set |
| `REPLACE_SECTION` | `REPLACE_RANGE` | set |
| `COVER` | `COVER` | set |

No title parsing, no prompt comparison, no inference from the mere
presence of a parent. `edit_kind` is the discriminator and the parent
link only separates ORIGINAL from GENERATE_AGAIN.

**A reference-conditioned generation with no parent is ORIGINAL**, and
falls out of the table for free: `reference_audio_id` appears nowhere in
it. A reference is input provenance, not derivation provenance — the
song was not made *from* another song.

**Duplicate Settings appears nowhere** because it creates nothing. It
navigates to Create with fields prefilled; no row exists until the user
submits, and what they submit is an ordinary create.

So Step 3's "if one explicit durable field is required, add the smallest
schema" is not reached. Phase 17 needs **no migration**; head stays at
`0012`.

## 3. Finding: deletion silently orphans descendants

`GenerationRepository.delete_generation` (repository.py:236-243) does
this to every child of the row being deleted:

```python
child.parent_generation_id = None
```

The docstring calls it deliberate — "Descendants are kept and re-pointed
to NULL. Deleting a take must never delete the takes made from it" — and
that reasoning is sound as far as it goes. Deleting a parent must not
destroy its children.

But the consequence is that **the derived song silently becomes an
ORIGINAL**. Its `edit_kind` still says `EXTEND` while its parent link is
gone, so it is a row claiming to be an extension of nothing. Under the
Phase 17 taxonomy that is not merely a missing edge; it is a
contradiction, and version history would render it as a root that says
"Extended".

This predates Phase 17 and is invisible today only because no lineage
exists yet to corrupt (see §4). It is the single most important thing
Phase 17 has to fix, and Step 20's recommended policy — reject deletion
of a row that still has children with a 409 — resolves it without
cascading and without re-parenting.

Note the interaction with Phase 16: blocking the delete also keeps the
generation's `reference_audio_id` in place, so a reference stays
protected for exactly as long as any song made from it exists. No change
to the cleanup grace policy is needed or wanted.

## 4. Finding: there is no lineage data at all

Every row in the live database is an ORIGINAL:

```
34 rows | edit_kind=NULL parent=false ref=false
 2 rows | edit_kind=NULL parent=false ref=true
------
36 total, 0 rows with parent_generation_id
```

The Extend, Replace Section and Cover features were each proved against
the engine directly during Phases 13B/13C/13D — through benchmark
scripts and the provider, not through the product's own create routes —
so no parent-linked row was ever written. The two reference-conditioned
rows are from Phase 15R and 15R-UI, and both correctly classify as
ORIGINAL.

This matters for Step 30. "Prefer existing real generations… search the
existing DB for known actual relationships created during prior phases"
cannot be satisfied: there are none. A real multi-level lineage
(A → B → C) has to be created, and each level is a real engine run of
roughly 90 seconds that cannot start until its parent has COMPLETED.
That is sequential and unavoidable.

## 5. Per-operation audit

### ORIGINAL
- **Parent** none · **Group** own · **Durable** absence of both discriminators
- **API** all fields present on `GenerationResponse`
- **UI** Song Detail renders it with no lineage context
- **Lost** nothing · **Ambiguity** none
- **Delete risk** becomes a parent-with-children the moment anything is derived from it

### GENERATE_AGAIN
- **Parent** set from the request · **Group** new · **Durable** parent set, `edit_kind` NULL
- **API** `parent_generation_id` exposed
- **UI** the button exists on Song Detail; the resulting child shows no relationship to its source
- **Lost** nothing durable
- **Ambiguity** **none for the taxonomy**, but worth stating: any client that supplies
  `parent_generation_id` on an ordinary create produces a GENERATE_AGAIN row. That is the
  correct reading — the user did derive it from that song — but it means the classification
  follows the request, not a dedicated endpoint.
- **Delete risk** as ORIGINAL

### EXTEND
- **Parent** set · **Group** new (`uuid4()` at route line 842) · **Durable** `edit_kind='EXTEND'`, plus `edit_start_seconds`/`edit_end_seconds`
- **API** exposed · **UI** no lineage display
- **Lost** nothing
- **Ambiguity** the group id is *new*, not inherited from the parent, so `generation_group_id`
  is not a lineage grouping. It groups siblings of one CREATE and nothing more. Lineage must
  be reconstructed from `parent_generation_id`, never from the group.
- **Delete risk** deleting the parent currently nulls this row's parent link (§3)

### REPLACE_SECTION
- **Parent** set · **Group** new · **Durable** `edit_kind='REPLACE_RANGE'` with both time bounds
- **API** exposed
- **UI** no lineage display; the replaced range is not shown anywhere after the fact
- **Lost** nothing durable
- **Ambiguity** the stored enum is `REPLACE_RANGE` while the product word is
  "Replace section" — a presentation mapping is required so the engine-adjacent name never
  reaches a user
- **Delete risk** as EXTEND

### COVER
- **Parent** set · **Group** new · **Durable** `edit_kind='COVER'`, `source_adherence`
- **API** exposed · **UI** no lineage display
- **Lost** nothing
- **Ambiguity** none. `EditKind.COVER`'s own docstring notes a cover is not an edit at all —
  it shares the column because it answers the same question, "how did this come from its
  parent?"
- **Delete risk** as EXTEND

### Reference-conditioned generation
- **Parent** none (unless independently derived) · **Durable** `reference_audio_id`
- **API** the id is accepted on create but **not returned** on `GenerationResponse`
- **UI** nothing shown after submission
- **Lost** the association is invisible to the client, which is why Phase 15R-UI reported
  Reuse Settings as PARTIAL
- **Ambiguity** none for lineage: it must classify as ORIGINAL
- **Delete risk** none to lineage; interacts with Phase 16 cleanup as described in §3

## 6. Projects are independent of lineage

`project_id` is set only by explicit assignment (`PATCH`, bulk-project,
or the create payload). No derivation path copies it from a parent:
Extend, Replace Section and Cover all omit it, so a child starts with no
project. Lineage and project membership are therefore orthogonal today,
and Step 22 is satisfied by leaving it alone and documenting it. A child
must not silently inherit or move.

## 7. What Phase 17 must build

| Step | Needed? | Why |
|---|---|---|
| Migration | **No** | §2 — taxonomy is derivable |
| `root_generation_id` column | **No** | parent traversal suffices; bounded depth guards cost |
| Lineage service + bounded traversal | Yes | with self-parent, cycle and depth defences |
| `GET /v1/generations/{id}/lineage` | Yes | UI must not reconstruct a graph from N requests |
| Presentation mapping | Yes | `REPLACE_RANGE` → "Replaced 0:45–1:00" |
| Version history UI | Yes | none exists |
| Delete policy → 409 | **Yes, and urgent** | §3 is a live provenance bug |
| Real multi-level lineage | Yes, must be created | §4 — none exists to reuse |

---

*Audit only. No schema, service, route or component was changed to
produce it.*
