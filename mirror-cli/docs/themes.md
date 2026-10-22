# Terminal themes

`/themes` opens the selector: Enter applies, Escape cancels. `/themes <name>`
selects directly, including during an active turn.

| Theme | Use |
| --- | --- |
| `auto` (default) | Match the terminal's light or dark background. |
| `dark` | Bright accents on dark surfaces. |
| `light` | Dark accents on light surfaces. |
| `terminal` | Standard ANSI colors and terminal-default backgrounds. |
| `mono` | No first-party colors; bold/reverse selection and text cues. |

## Detection and overrides

Mirror detects the background at interactive startup without changing it.
Detection can take up to half a second; if unavailable, `auto` uses `COLORFGBG`,
then falls back to `dark`. Restart Mirror after changing terminal colors.
Set `MIRROR_TERMINAL_BACKGROUND=light` or `dark` to override and skip detection.
A saved explicit theme remains selected; use `/themes auto` to follow the terminal.

`NO_COLOR` (even empty) or `TERM=dumb` forces monochrome without changing the
saved preference. `COLORTERM=truecolor` or `24bit` enables RGB colors;
`PROMPT_TOOLKIT_COLOR_DEPTH` overrides color depth. Otherwise, the terminal
capabilities determine the depth.

## Saved preference

`/themes` saves the choice in `~/.mirror/config.json`:

```json
{"extensions": {"builtin/themes": {"config": {"name": "auto"}}}}
```

Missing or invalid names fall back to `auto`. Disabling `builtin/themes` removes
`/themes` and restores the default. Failed saves leave the preference unchanged.

Changes apply immediately in the current workspace; other workspaces/processes
refresh within about five seconds. Conversation output repaints when foreground
and idle; scrolled-off terminal history may retain old colors.

Code blocks, command approvals, and diffs use syntax colors suited to the palette.
Custom themes inherit their base's syntax colors. Tool ANSI output, workspace-tag
colors, and extension color literals are not remapped.
