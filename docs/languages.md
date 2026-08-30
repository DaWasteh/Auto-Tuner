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

## Custom packs

Choose **Open language folder** once. AutoTuner opens
`~/.autotuner/languages` and creates
`custom-language-template.json` if it does not already exist. The template is a
complete copy of the English catalogue and is never overwritten after creation.

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
- A pack is limited to 2 MiB and 2,000 strings.

Select **Custom language pack…**, choose the JSON file, and confirm replacement
if that custom ID already exists. AutoTuner validates a staged copy before it
atomically replaces anything in the language folder. Invalid JSON, schemas, IDs,
or string values are rejected without changing the active language.

Built-in packs under `assets/languages` are read-only application resources.
Custom packs under `~/.autotuner/languages` survive source updates and compiled
binary swaps.
