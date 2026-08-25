# Decision 0011: V4 A02 zero-shot generalization protocol

## Status

Complete on 2026-08-24. The frozen V4 policy finished 10 of 20 deterministic
A02 episodes without retraining. This is evidence of partial zero-shot transfer
under the shared centerline observation, but not reliable generalization.

## Result

The pre-registered 20-episode run used the exact frozen V4 checkpoint and 6x
simulation speed. It produced:

- finish rate: `10/20` (`50%`)
- best / mean / worst finish: `20.110 s` / `22.461 s` / `24.900 s`
- terminal causes: `10` finishes, `10` verified stuck truncations, `0` falls,
  and `0` timeouts
- steering oscillation: `13/20` episodes
- upside-down behavior: `8/20` episodes, totaling `34.0 s`
- mean p95 / maximum absolute lateral deviation: `10.128` / `20.379` units
- mean significant direction reversals: `28.75`

The result is mixed. V4 transferred enough of its learned response to the
engineered track-relative observations to complete a geometrically different
track half the time, and its best zero-shot finish was only `3.58 s` behind the
live-replayed Nadeo author reference used to build the centerline. However, the
other half of the episodes stopped moving long enough to trigger the unchanged
2.1-second stuck detector, and inversion/oscillation remained common. This is
therefore not evidence that the policy is generally reliable outside A01.

## Standalone generalization finding

Stated plainly: V4 achieves near-human-PB performance on A01 but generalizes
poorly to A02 zero-shot. On A01 it finished `98/100` scale-validation episodes
and reached `24.900 s`, only `0.400 s` behind the owner's real `24.5 s` PB. The
same frozen policy, supplied with the same kind of track-relative centerline
observations on A02, finished only `10/20`; eight episodes inverted and ten
ended through the stuck detector.

This is evidence that V4 has substantially overfit to A01's specific geometry
rather than learning general driving. Its ten A02 finishes show some transferable
control, so it is not merely replaying a fixed A01 action sequence, but the drop
from near-perfect A01 reliability to 50% A02 reliability is too large to call
the learned behavior track-general. Do not retrain or fine-tune this checkpoint
on A02: preserving the frozen zero-shot measurement is the point of this result.
Any later A01+A02 training is a new multi-track experiment, not a correction to
Decision 0011.

The evaluation artifacts are preserved as:

- summary: `runs/reward_v4_a02_zero_shot/evaluation_summary.json`, SHA-256
  `D05AF1725B693CC293F3635A03EDB7AC63DB0D2AFBE48F4B0D813C46A2BD5694`
- action audit: `runs/reward_v4_a02_zero_shot/evaluation_actions.jsonl`,
  SHA-256
  `C7EDDCEA1FC675C7FBA33F83D289D36F1741824C4C07B97B6BD374CA62AD96DD`
- all 20 preserved inputs: `artifacts/replays/reward_v4_a02_zero_shot/`

Clean, overlay-free H.264 captures were replayed from the preserved inputs, not
rerolled from the policy. The closest successful episode to the 22.461-second
finish mean is episode 4 at 22.170 seconds:

- best, episode 8: SHA-256
  `AADC3579FBD5FCE461780FCB53E5ACCE2D73F8D4DCB2B865057EAC8214ECFF82`
- closest to mean, episode 4: SHA-256
  `7FB81BE920977AB4AAD405F1E0017A8D7BDC8EDA721CD74621086ADD74F62A75`
- worst finish, episode 20: SHA-256
  `0C12A88E15B0BF31B4E7D5839FC5BBE09BC047475AFC1E33C51328312F4228DA`

## Question

Does the finished V4 policy transfer at all from A01-Race to a different,
similarly introductory TMNF track without retraining or adaptation?

## Decision

Use the built-in `A02-Race` as the second track. It is adjacent to A01 in the
Nations White campaign but has different turn geometry. Run the already-finished
V4 checkpoint for 20 deterministic episodes at 6x simulation speed. Do not train,
fine-tune, select checkpoints, or change actions based on A02 results.

The model checkpoint is frozen as:

- `checkpoints/reward_v4/final_model.zip`
- SHA-256 `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`
- model timestep `2,002,944`

Build A02's reference path from the installed official Nadeo author replay, not
from an agent attempt. Pin the installed source files before extraction:

- `A02-Race.Challenge.Gbx`: SHA-256
  `DCBB1376DCBD10A6018E26D6991EF0155717FA27A57AED27236B95A5876A9D80`
- `A02-Race.Replay.gbx`: SHA-256
  `7546E19CE9CA0D36E074406256A547A256F20B23602985085E98082EE7D178E3`

Extract the author inputs with the TMInterface author's public `gbxtools`
conversion semantics and checksum-pinned `pygbx==0.3` parser. This fallback is
required because the installed TMInterface 2.2.1 direct-file `dump_inputs`
command rejects even a replay that the same client saved and extracted through
its UI. Replay the extracted inputs at 100 ms telemetry intervals, require a
real finish and a monotonic race clock, then resample the driven positions at
the same 5-unit spacing used for A01. Preserve the extracted inputs, raw
telemetry, reference path, hashes, and provenance manifests.

The evaluation will retain the same action audit, replay retention, finish/fall/
stuck classification, lateral-deviation metrics, and oscillation detection used
for V4 on A01. Results are descriptive; there is no pass gate and no A02 retry
based on outcome.

## Interpretation boundary

This is zero-shot **policy transfer over the same engineered observation type**.
The policy receives A02-relative centerline progress, heading, lateral offset,
and look-ahead geometry just as it received A01-relative values during training.
It is not raw-pixel generalization and it is not navigation without a supplied
reference line. A failure still demonstrates A01 specialization; a success only
supports transfer under this explicitly shared representation.

## Why A02

A02 is a stronger controlled comparison than a distant or advanced track:

- beginner campaign placement limits the difficulty jump;
- its geometry differs enough to expose A01-specific steering behavior;
- the installed official replay supplies reproducible, first-party geometry;
- no external leaderboard run or manually chosen racing line is required.

No V5 checkpoint will be used for this measurement. V5 remains the separate,
single-variable steering-smoothness experiment requested by the owner.

## Resolved setup record

The installed challenge and replay hashes matched the pre-registration. The
exact UI-observed paths on this Steam/TMUF-compatible profile are:

- TrackMania's Replay browser reads
  `Documents\TrackMania\Tracks\Replays`, not the nominal TMNF
  `Documents\TmForever\Tracks\Replays` folder.
- TMInterface 2.2.1's **Save Current Replay** UI writes to
  `Documents\TrackMania\Tracks\Replays\TMInterface` because `replay_folder` is
  `TMInterface`.
- The saved local round-trip replay was
  `a01v4roundtrip.Replay.Gbx`, 42,644 bytes, SHA-256
  `2A978D8AADE6BC9EB04701BD2C7ACE4295FB2148DD67E0096C5C853D8BE2C8E3`.
- The adjacent **Get Inputs** UI action successfully copied 8,128 characters of
  inputs from that replay and ended them with neutral controls at 24.95 s.

This proves the replay data and UI extraction path are valid. In contrast, the
bridge tried the installed 2.2.1 direct replay-file command against both the
pinned A02 replay and TMInterface's own local round-trip replay using absolute,
Replay-folder-relative, game-root-relative, and Scripts-folder paths; quoted
and unquoted paths; one- and two-argument forms; and lowercase `.gbx` staging.
Representative commands were:

```text
dump_inputs "C:\Users\Vardhan\Documents\TrackMania\Tracks\Replays\TMInterface\a01v4roundtrip.Replay.Gbx" a01v4roundtrip_direct.txt
dump_inputs a01v4roundtrip.Replay.Gbx
dump_inputs Tracks/Replays/TMInterface/a01v4roundtrip.Replay.Gbx
```

Every direct-file form produced TMInterface's own error, `Failed to extract
inputs, the file may not be a valid replay or exist.` The in-game Validate
button is also disabled for the installed Nadeo replay, which TMInterface's
guide identifies as an online/protected-replay limitation. There is therefore
no verified direct-file bridge syntax to preserve for this build: the verified
2.2.1 workflow is the managed Replay UI, while protected A02 requires the
official external-parser fallback.

The fallback found four finishing `alinoa` ghosts in the pinned replay and
selected the fastest one deterministically:

- ghost index: `0`
- recorded race time: `16,250 ms`
- control entries: `53`
- extracted input bytes: `544`
- extracted input SHA-256:
  `E476F390F8FA6ADF3F8C7488F366D6118474883D1F6B18494CFEE812A4CE7142`

Live playback then exposed a separate map-search issue. `map
A02-Race.Challenge.Gbx` searches the indexed user folder, not Steam's built-in
campaign tree. The checksum-pinned challenge must be present at
`Documents\TrackMania\Tracks\Challenges\A02-Race.Challenge.Gbx` before TMNF
starts; the capture/evaluation tools now install it safely and refuse to
overwrite a different user file. A fresh launch is required for TMNF to index a
new copy.

The unchanged extracted inputs finished A02 live at `16,530 ms`, producing 167
finite, monotonic 100 ms samples over 845.114 units. The 280 ms difference from
the embedded ghost time comes from reducing sub-10 ms replay events to
TMInterface's 10 ms input-command grid; this reference is used for geometry,
not as an author-time claim. Resampling produced 171 points at 5-unit spacing:

- telemetry SHA-256:
  `1270BFD662A5539ED3A32EB6F309355CA8921701BB3762A8A3E5D1E037D2081F`
- reference CSV SHA-256:
  `B6570AFD0B2FD87D0E0BB9286D94867A0BB7C19DEE2BEC8F49ADED089BF316AE`

Several launcher hardening findings were kept separately in git history: a TCP
readiness probe was consuming a real `python_link.as` client connection, fresh
launches injected focus-dependent Enter presses at the main menu, and a proposed
queued-handshake plugin recovery did not fix the live failure. The unsuccessful
plugin change was removed from the active installed copy. A later live capture
also showed that a visible loading window and open listener do not yet mean
TMInterface has left `StartUp`; fresh launches now require ten continuous
seconds of observed window/listener readiness before a bridge client connects.
These issues are tooling results, not A02 policy performance.

The measurement used A02's centerline without substituting A01 geometry,
retraining on A02, or selecting a checkpoint based on A02 behavior. The frozen
checkpoint hash matched the pre-registration before evaluation.
