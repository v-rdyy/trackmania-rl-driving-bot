# Vendored TMInterface Python bridge

`python_link.as` is vendored from `dersiwi/trackmania-gym` at commit `1d066ee742e736f6388818df2e07a4a89329f598`.

- Source: https://github.com/dersiwi/trackmania-gym/blob/1d066ee742e736f6388818df2e07a4a89329f598/src/trackmania_gym/game_interaction/plugin/python_link.as
- Upstream file SHA-256: `63305AB927B0D199DA1DB7D011F7720C5F6397BF93C2EE2102C5AEC1CD224365`.
- Vendored file SHA-256: `A47CCE075B3020BE9234C95135A170A58C1B950811B69963D63831813DC6366E`. The only byte-level difference is an added final LF line terminator; removing that byte reproduces the upstream hash.
- License: GNU General Public License v3.0; see `LICENSE` in this directory.
- Plugin metadata names Agade as the author and identifies the plugin as Python Link `0.1`.

The bridge listens on `127.0.0.1` at the TMInterface `custom_port` supplied when the game starts. Keep it loopback-only. This project uses port `8478` for the first Phase 0 instance.
