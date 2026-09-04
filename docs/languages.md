# AutoTuner language packs

AutoTuner v5.4.1 includes these validated interface packs:

- English (UK), the fallback and default
- Deutsch
- Nederlands
- Svenska
- 日本語
- Français
- Ελληνικά
- Polski
- Русский (new in v5.4.1)

Open the persistent **⋯** toolbar and use **Language** to change the interface
immediately. Missing translations always fall back to English (UK), so an older
custom pack remains usable when a later AutoTuner version adds controls.

## What is translated (v5.4.1)

Every built-in pack carries 427 strings:

- toolbar, launch-options, server, and settings labels (the original 83);
- the two-level hover help of every control: the **In short:** and
  **Technical details:** section labels and both texts;
- the Expert panel sections and labels, the performance-test dialog, the OCR
  dialog, the performance analysis window, and the path-manager dialog;
- message boxes (update, folder and launch errors, Expert copy/paste);
- the model context menu, the **★ Favourites** tree section, and the
  list/tree row tooltips (favourite state, vision, draft, thinking, tool use);
- the performance-target descriptions shown in the Performance tooltip and in
  the performance-test dialog;
- the metric explanations of the performance analysis.

Hover help is stored as plain text. `localization.setting_tooltip_html()`
builds the HTML that Qt shows; `LanguageManager.translate_tooltip()` takes it
apart again, translates the section labels and both texts, and rebuilds it.
Lines the application composes at runtime keep their dynamic part: a pack
translates `Active build:` and `Resolved path:` once, and the tier bullet list
`• balanced: …` is translated through the tier description. Texts that Qt
widgets cannot retranslate live (item tooltips, dynamic labels, message boxes)
go through `localization.translate()` when they are created, and switching the
language rebuilds the model list.

`test_localization.py` extracts every help constant from `qt_launcher.py` with
the `ast` module (`_setting_tooltip(...)` arguments, `_tr(...)` arguments, the
model-row tooltip constants, the metric help, the fork tooltip constants, and
the performance-target descriptions) and fails when any built-in pack lacks a
key or leaves a long text untranslated. Adding a control with hover help
therefore means adding its two texts to all nine packs.

## Model profile explanations (v5.3.9)

Every model profile under `settings/*.yaml` carries a `notes` explanation that
the configuration preview shows for the selected model. Since v5.3.9 the YAML
text is the English canonical version, and language packs may translate it
through an optional `profile_notes` object keyed by the profile file name:

```json
{
  "schema_version": 1,
  "id": "de-DE",
  "name": "Deutsch",
  "locale": "de-DE",
  "strings": { "⚙ Settings": "⚙ Einstellungen" },
  "profile_notes": {
    "gemma-4.yaml": "Gemma ist empfindlich gegenüber repeat_penalty > 1.0. …"
  }
}
```

- Switching the language re-renders the preview, so the explanation follows
  the interface language immediately.
- Lookup order: the active pack, then the English (UK) pack, then the YAML
  `notes` text. A pack may translate any subset of profiles.
- All nine bundled packs translate every shipped profile (76 in v5.4.1).
  The YAML `notes` text is the English source; a custom pack may translate any
  subset through `profile_notes`.
- Keys must be profile file names (`name.yaml`, letters, digits, `.`, `_`, `-`);
  values are non-empty strings up to 20,000 characters; at most 500 entries.
- The CLI/TUI prints the English YAML text.

## Custom packs

Choose **Open language folder** once. AutoTuner opens
`~/.autotuner/languages` and creates
`custom-language-template.json` if it does not already exist. The template is a
complete copy of the English catalogue, including the hover help and
`profile_notes`, and is never overwritten after creation.

A pack is UTF-8 JSON:

```json
{
  "schema_version": 1,
  "id": "example",
  "name": "Example language",
  "locale": "xx-XX",
  "strings": {
    "⚙ Settings": "⚙ My translation",
    "Language:": "My language label:",
    "In short:": "Briefly:",
    "Changes AutoTuner's interface language immediately.": "My hover text"
  },
  "profile_notes": {
    "qwen3_8.yaml": "My explanation for Qwen3.8"
  }
}
```

Rules:

- `id` must be 1–64 characters and contain only letters, digits, `.`, `_`, or
  `-`; it must start with a letter or digit.
- `name` is the native display name shown in the dropdown.
- `locale` documents the intended locale.
- `strings` maps the exact English (UK) source text to its translation. Entries
  may be omitted; English is then used for those strings. Hover-help texts are
  keyed by their plain English text, not by the generated HTML.
- `profile_notes` is optional and follows the rules above.
- A pack is limited to 2 MiB and 2,000 strings.

Select **Custom language pack…**, choose the JSON file, and confirm replacement
if that custom ID already exists. AutoTuner validates a staged copy before it
atomically replaces anything in the language folder. Invalid JSON, schemas, IDs,
string values, or profile notes are rejected without changing the active
language.

Built-in packs under `assets/languages` are read-only application resources.
Custom packs under `~/.autotuner/languages` survive source updates and compiled
binary swaps.
