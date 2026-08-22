# Vendored TMInterface Python bridge

`python_link.as` is vendored from `dersiwi/trackmania-gym` at commit `1d066ee742e736f6388818df2e07a4a89329f598`.

- Source: https://github.com/dersiwi/trackmania-gym/blob/1d066ee742e736f6388818df2e07a4a89329f598/src/trackmania_gym/game_interaction/plugin/python_link.as
- Upstream file SHA-256: `63305AB927B0D199DA1DB7D011F7720C5F6397BF93C2EE2102C5AEC1CD224365`.
- Initial vendored SHA-256: `A47CCE075B3020BE9234C95135A170A58C1B950811B69963D63831813DC6366E`. At that point, the only byte-level difference from upstream was an added final LF line terminator.
- Current vendored SHA-256: `3B34057EDA19C69E43F2858F723773EFB257A7F425DE4C631A1018416383B7C1`.
- Local hardening: the plugin binds the configured loopback port during `Main()` instead of depending entirely on a queued startup callback, defaults to Phase 0 port `8478`, reports listen success/failure, safely handles a failed bind, and can rebind after a later `custom_port` change when its queue callback runs. While idle without a client, it waits up to 100 ms for a connection; the upstream zero-timeout poll and a 1 ms poll did not dequeue established clients on the verified TMInterface 2.2.1 runtime. While a client is connected, race/control traffic prevents replacement polling. After five seconds without client traffic, the plugin performs one bounded 100 ms accept so a newer connection can replace a socket left stale by an abruptly terminated Python process, without adding per-frame stalls during active capture. It also services pending control messages outside synchronous race callbacks so graceful shutdown works from menus. Python-supplied console commands run through `CommandListProcessOption::ExecuteImmediately`, following the 2.2.1 API guidance instead of relying on the legacy queued `ExecuteCommand` default.
- Phase 1 protocol extension: message ID `24` carries signed 32-bit steer and gas values. The plugin constrains both to TMInterface's documented analog range of `-65536..65536`, clears legacy digital driving inputs, and applies `InputType::Steer` plus `InputType::Gas`. Existing message IDs and the Phase 0 digital packet remain unchanged.
- License: GNU General Public License v3.0; see `LICENSE` in this directory.
- Plugin metadata names Agade as the author and identifies the plugin as Python Link `0.1`.

The bridge listens on `127.0.0.1` at the TMInterface `custom_port` supplied when the game starts. Keep it loopback-only. This project uses port `8478` for the first Phase 0 instance.
