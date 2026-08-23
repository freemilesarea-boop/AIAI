# Multi-window coverage, at one tensor shape

Phase 36 trained on the first two minutes of every track. Not because
two minutes was enough, but because Metal keeps an allocator working set
per tensor shape: a dataset with 24 different latent lengths reached 29
GiB where four tracks at the same maximum length peaked at 9.4, and the
run died at the same step four times. One fixed length made it work.

The cost was severe and easy to miss. Of 128 authorised tracks, **122
are longer than 120 seconds** — median 190 s, longest 377.5 s. The model
never saw a chorus that arrived late, a bridge, or an outro.

Phase 37 removes the cost without giving back the fix.

## The representation

Every window is 3000 latent frames. The rate was measured from the real
pipeline rather than taken from documentation — a 178.80 s track
preprocesses to exactly 4470 frames — so 3000 frames is exactly **120.0
seconds**, and a 120.0 s clip comes back as exactly 3000 frames.

How many windows a track yields follows from one rule rather than a
table of guesses: *consecutive windows may overlap by at most half their
length*. Spread `n` windows evenly and they sit `(frames - 3000) /
(n - 1)` apart, so `n` is admitted only while that spacing is at least
1500 frames. The thresholds fall out:

| track length | windows | positions |
|---|---|---|
| < 120 s | 0 — too short, and never padded | — |
| 120–180 s | 1 | START |
| 180–240 s | 2 | START, END |
| 240–300 s | 3 | START, MIDDLE, END |
| ≥ 300 s | 4 (capped) | START, EARLY_MIDDLE, LATE, END |

Six tracks in the library are shorter than one window (81.6 s to 118.6
s). They are excluded and *reported* rather than padded: a padded frame
is content the recording does not have, and a loss computed over it is a
loss over silence somebody inserted.

Windows are materialised as **audio**, cut on exact frame boundaries
with the standard library's `wave` module, and then run through the real
ACE-Step preprocessing pipeline. The source is 16-bit PCM at 48 kHz and
one latent frame is 1920 samples, so a window is a byte-range copy — no
decode, no resample, no re-encode.

## Deterministic, not a random crop

A window is a pure function of the track digest, the experiment seed and
the window index. There is no per-epoch random crop, because an
experiment whose data changes every epoch cannot be compared with
another one. The same manifest always describes the same audio, and the
manifest carries its own digest.

## Split tracks first. Always.

Windows are planned **inside** each split, never across the library.
Window a library and split the windows afterwards and two views of one
recording can land on opposite sides of a held-out boundary; the summary
table looks identical and the measurement is worthless.
`plan_windows` takes the tracks of a single split by design, and
`test_window_split_isolation.py` demonstrates both that our order is
safe and that the wrong order genuinely separates recordings.

## Track weighting

A four-window track and a one-window track are one recording each.
Weighting windows equally would give the long one four times the
influence for nothing but its length, so each window carries `1 /
windows_in_its_track` and every track sums to one.

**Known bias, stated plainly.** These weights are computed and recorded
in the window manifest, and the installed trainer's `PreprocessedDataModule`
does not consume a per-sample weight — it iterates the directory. So in
this phase the weights are *evidence*, not enforcement: training sees
each window once per epoch, and a four-window track does contribute four
times as many gradient steps as a one-window track. The distribution is
mild (128 windows over 64 tracks, mean 2.0, max 4) and it is written
down rather than glossed. Enforcing it needs a weighted sampler in the
loader, which is a trainer change and not this phase's variable.

## The capacity gap Phase 36 exposed

The old capacity model compared *sequence length* and nothing about
shape diversity. Phase 36's request and its profile agreed on
`latent_length` at every step of a failure that was entirely about
having 24 of them.

`MemoryProfileIdentity` now carries `latent_shape_count`, and it is
compared **exactly** — not monotonically, because a profile measured
over one shape says nothing about a run over twenty in either direction.
Demonstrated live: the same stored profile that qualifies Phase 37's
fixed-shape dataset refuses the exact 24-shape configuration Phase 36
ran, with the reason named.

## What the run produced

| | |
|---|---|
| Outcome | `COMPLETED` |
| Training signal | **`VALID_SIGNAL`** |
| Generalization signal | **`HELD_OUT_LOSS_IMPROVED`** |
| Steps | 320 of a 320 ceiling (module maximum 600), two segments of 160 |
| Wall clock | 4933 s |
| Train windows | 128 from 64 tracks (mean 2.0, max 4) |
| Unique latent shapes | **[3000]** |
| Train loss | first 1.6654, last 0.9627, min 0.9564, max 2.0367, finite ratio 1.0 |
| Held-out loss | first 1.4022, last 1.0430, min 1.0430, over 10 measurements on 8 tracks |
| Gradients | 320 of 320 finite, 320 non-zero, mean norm 0.3725 |
| Adapter | **384 of 384** tensors changed, 11,010,048 parameters |
| Base model | preserved — file digest identical before and after |
| Resume | epoch 5 to 10, step 160 to 320, exit 0 |
| Checkpoint provenance | both segments verified |

### Memory: the thing Phase 36 died of

device memory went from 4.57 GiB to 4.58 GiB between the first and last quarter of the run (+0.1%), peaking at 4.58 GiB.

Across all 320 steps the device held a flat 4.58 GiB. Phase
36's variable-shape dataset reached 29 GiB and died at step 9 of every
attempt. The early safety segment measured the same flat profile over 64
steps before the full budget was committed, which is what the gate is
for.

### Phase 36 versus Phase 37

| | Phase 36 | Phase 37 |
|---|---|---|
| Train tracks | 24 | 64 |
| Train samples | 24 windows (1 each) | 128 windows (mean 2.0) |
| Song coverage | first 120 s only | up to 4 positions per track |
| Validation tracks | 4 | 8 |
| Optimizer steps | 120 | 320 |
| Held-out loss | 1.4724 to 1.1278 (-23.41%) | 1.4022 to 1.0430 (-25.62%) |
| Peak device memory | 29 GiB, OOM | 4.58 GiB, flat |

**The two absolute losses are NOT DIRECTLY COMPARABLE.** Different
validation tracks, different counts, and a different window length go
into them. What can be compared is the *shape*: both runs moved held-out
loss down, and Phase 37 did it on twice the held-out material.

That is not a claim that Phase 37 is the better model. Nothing here can
decide that.

## Listening evaluation

Two sets, each generated three times — untouched base, Phase 36 adapter,
Phase 37 adapter — with identical prompt, seed, duration and step count,
and the previous adapter unloaded before the next was attached:

- **`data/evaluation/exp37/abc/`** — 8 prompt/seed sets drawn from the
  Phase 37 evaluation split and nowhere else, 30 s each. 48 files, all
  decoding, none silent.
- **`data/evaluation/exp37/vocal/`** — the same three Pop/R&B songs the
  operator listened to after Phase 36, same lyrics and seeds. 18 files,
  all decoding, none silent.

## What is still not established

- Whether either adapter makes better music. Nobody has listened to the
  three-way set yet.
- Whether Phase 37 improves on Phase 36. The losses are not comparable
  and the listening test has not been done.
- Whether anything converged. 320 steps cannot answer that.
- Whether the held-out improvement survives another seed or split.
- 48 authorised tracks were deliberately never seen, and 6 more are
  shorter than one window and were excluded.
