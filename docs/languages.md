# AutoTuner language packs

AutoTuner v5.3.4 includes these validated interface packs:

- English (UK), the fallback and default
- Deutsch
- Nederlands
- Svenska
- 日本語
- Français
- Ελληνικά
- Polski

Open the persistent **⋯** toolbar and use **Language** to change the interface
immediately. Missing translations always fall back to English (UK), so an older
custom pack remains usable when a later AutoTuner version adds controls.

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
- All eight bundled packs translate every shipped profile (76 in v5.3.9).
  The YAML `notes` text is the English source; a custom pack may translate any
  subset through `profile_notes`.
- Keys must be profile file names (`name.yaml`, letters, digits, `.`, `_`, `-`);
  values are non-empty strings up to 20,000 characters; at most 500 entries.
- The CLI/TUI prints the English YAML text.

## Custom packs

Choose **Open language folder** once. AutoTuner opens
`~/.autotuner/languages` and creates
`custom-language-template.json` if it does not already exist. The template is a
complete copy of the English catalogue, including `profile_notes`, and is never
overwritten after creation.

A pack is UTF-8 JSON:

```json
{
  "schema_version": 1,
  "id": "example",
  "name": "Example language",
  "locale": "xx-XX",
  "strings": {
    "⚙ Settings": "⚙ My translation",
    "Language:": "My language label:"
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
  may be omitted; English is then used for those strings.
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
