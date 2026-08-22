# Pre-V3 backup

Status: Complete; V3 remains unstarted

Date: 2026-08-22

The project had no configured Git remote and the GitHub CLI was not installed,
so the approved checksum-pinned local bundle fallback was used instead of a
GitHub Release.

`artifacts/backups/trackmania_v2_pre_v3_e7455fd.zip` contains:

- a Git bundle with every ref through source commit
  `e7455fde772b91f747d101b082e119b1bbec388a`;
- all reward-V2 checkpoints and training/evaluation run evidence;
- every reward-V2 TensorBoard event file;
- the full V0-V2 replay catalog, input replays, manifests, and round-trip
  telemetry/results;
- the corrected gas-direction probe evidence;
- an internal manifest with the size and SHA-256 of each evidence file.

The archive contains 105 evidence files totaling 19,566,341 bytes before ZIP
compression. Its final size is 6,865,462 bytes. Python's ZIP integrity test
passed, and an independent `Get-FileHash` result matched the adjacent checksum
file.

Archive SHA-256:

`16B8B989E8A61E89BB24BBC0BAC567381E2C55DF2C5C039924F531177E25FA93`

The adjacent
`artifacts/backups/trackmania_v2_pre_v3_e7455fd.zip.sha256` file records the same
digest. The backup directory is intentionally ignored by Git so the binary
archive is not mistaken for source history. Do not begin V3 training unless this
archive and checksum remain present and the owner has separately approved V3's
pre-registered hypothesis and thresholds.
