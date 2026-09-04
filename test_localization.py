from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from localization import (
    DEFAULT_LANGUAGE_ID,
    TOOLTIP_SUMMARY_LABEL,
    TOOLTIP_TECHNICAL_LABEL,
    LanguageManager,
    LanguagePackError,
    setting_tooltip_html,
)


ROOT = Path(__file__).resolve().parent
BUILTIN_LANGUAGES = ROOT / "assets" / "languages"


def test_builtin_language_packs_are_complete_and_grammatically_named(tmp_path) -> None:
    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    packs = manager.available()
    assert [pack.qualified_id for pack in packs] == [
        "builtin:en-GB",
        "builtin:de-DE",
        "builtin:nl-NL",
        "builtin:sv-SE",
        "builtin:ja-JP",
        "builtin:fr-FR",
        "builtin:el-GR",
        "builtin:pl-PL",
        "builtin:ru-RU",
    ]
    assert [pack.name for pack in packs] == [
        "English (UK)",
        "Deutsch",
        "Nederlands",
        "Svenska",
        "日本語",
        "Français",
        "Ελληνικά",
        "Polski",
        "Русский",
    ]
    english_keys = set(manager.packs[DEFAULT_LANGUAGE_ID].strings)
    assert len(english_keys) >= 400
    assert all(set(pack.strings) == english_keys for pack in packs)
    assert not manager.errors

    expected = {
        "builtin:de-DE": ("⚙ Einstellungen", "Sprache:"),
        "builtin:nl-NL": ("⚙ Instellingen", "Taal:"),
        "builtin:sv-SE": ("⚙ Inställningar", "Språk:"),
        "builtin:ja-JP": ("⚙ 設定", "言語:"),
        "builtin:fr-FR": ("⚙ Paramètres", "Langue :"),
        "builtin:el-GR": ("⚙ Ρυθμίσεις", "Γλώσσα:"),
        "builtin:pl-PL": ("⚙ Ustawienia", "Język:"),
        "builtin:ru-RU": ("⚙ Настройки", "Язык:"),
    }
    for pack_id, (settings, language) in expected.items():
        manager.select(pack_id)
        assert manager.translate("⚙ Settings") == settings
        assert manager.translate("Language:") == language


def test_custom_pack_import_is_validated_namespaced_and_atomic(tmp_path) -> None:
    user_dir = tmp_path / "languages"
    manager = LanguageManager(BUILTIN_LANGUAGES, user_dir)
    custom = tmp_path / "pirate.json"
    custom.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "pirate",
                "name": "Pirate",
                "locale": "en-XP",
                "strings": {"⚙ Settings": "⚙ Ship settings"},
            }
        ),
        encoding="utf-8",
    )
    imported = manager.import_pack(custom)
    assert imported.qualified_id == "user:pirate"
    assert (user_dir / "pirate.json").is_file()
    manager.select(imported.qualified_id)
    assert manager.translate("⚙ Settings") == "⚙ Ship settings"
    assert manager.translate("Language:") == "Language:"

    custom.write_text(custom.read_text(encoding="utf-8").replace("Ship", "Crew"), encoding="utf-8")
    with pytest.raises(FileExistsError):
        manager.import_pack(custom)
    replaced = manager.import_pack(custom, replace=True)
    manager.select(replaced.qualified_id)
    assert manager.translate("⚙ Settings") == "⚙ Crew settings"
    assert not list(user_dir.glob("*.tmp"))


def test_invalid_custom_pack_is_ignored_without_disabling_builtins(tmp_path) -> None:
    user_dir = tmp_path / "languages"
    user_dir.mkdir()
    (user_dir / "broken.json").write_text('{"schema_version": 1}', encoding="utf-8")
    manager = LanguageManager(BUILTIN_LANGUAGES, user_dir)
    assert manager.current_id == DEFAULT_LANGUAGE_ID
    assert len(manager.available()) == 9
    assert manager.errors and "broken.json" in manager.errors[0]
    with pytest.raises(LanguagePackError):
        manager.import_pack(user_dir / "broken.json", replace=True)


def test_custom_template_is_created_once_from_english_catalogue(tmp_path) -> None:
    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    path = manager.ensure_custom_template()
    first = path.read_bytes()
    payload = json.loads(first.decode("utf-8"))
    assert payload["id"] == "my-language"
    assert payload["strings"]["⚙ Settings"] == "⚙ Settings"
    path.write_text("user edit", encoding="utf-8")
    assert manager.ensure_custom_template().read_text(encoding="utf-8") == "user edit"


def test_live_widget_retranslation_always_uses_original_source(tmp_path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    assert app is qt_widgets.QApplication.instance()
    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    root = qt_widgets.QWidget()
    layout = qt_widgets.QVBoxLayout(root)
    button = qt_widgets.QPushButton("⚙ Settings")
    label = qt_widgets.QLabel("Language:")
    layout.addWidget(button)
    layout.addWidget(label)

    manager.select("builtin:de-DE")
    manager.apply_to(root)
    assert button.text() == "⚙ Einstellungen"
    assert label.text() == "Sprache:"

    manager.select("builtin:fr-FR")
    manager.apply_to(root)
    assert button.text() == "⚙ Paramètres"
    assert label.text() == "Langue :"

    # Application-owned dynamic changes replace the remembered source rather
    # than being mistaken for a translation from the prior pack.
    button.setText("Close")
    manager.apply_to(root)
    assert button.text() == "Fermer"
    root.close()


def test_profile_notes_follow_the_interface_language(tmp_path) -> None:
    from settings_loader import load_profiles

    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    assert not manager.errors
    profiles = load_profiles(ROOT / "settings")
    files = {profile.source_file for profile in profiles}
    english = manager.packs[DEFAULT_LANGUAGE_ID]
    # Every bundled pack explains every shipped profile.
    for pack in manager.available():
        assert set(pack.profile_notes) == files, pack.qualified_id
    # The YAML text is the English canonical version.
    for profile in profiles:
        assert " ".join(profile.notes.split()) == english.profile_notes[profile.source_file]
    gemma = next(profile for profile in profiles if profile.source_file == "gemma-4.yaml")

    manager.select(DEFAULT_LANGUAGE_ID)
    assert manager.profile_notes(gemma).startswith("Gemma is sensitive")
    manager.select("builtin:de-DE")
    assert manager.profile_notes(gemma).startswith("Gemma ist empfindlich")
    assert manager.profile_notes("gemma-4.yaml").startswith("Gemma ist empfindlich")
    manager.select("builtin:ja-JP")
    assert manager.profile_notes(gemma).startswith("Gemma は")
    # Unknown profiles fall back to the supplied YAML text.
    assert manager.profile_notes("does-not-exist.yaml", "yaml fallback") == "yaml fallback"
    assert manager.profile_notes(None, "plain") == "plain"

    template = json.loads(manager.ensure_custom_template().read_text(encoding="utf-8"))
    assert template["profile_notes"]["gemma-4.yaml"].startswith("Gemma is sensitive")


def test_profile_notes_are_validated_in_custom_packs(tmp_path) -> None:
    user_dir = tmp_path / "languages"
    manager = LanguageManager(BUILTIN_LANGUAGES, user_dir)
    custom = tmp_path / "notes.json"
    custom.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "id": "notes",
                "name": "Notes",
                "locale": "xx-XX",
                "strings": {},
                "profile_notes": {"gemma-4.yaml": "Custom Gemma explanation"},
            }
        ),
        encoding="utf-8",
    )
    imported = manager.import_pack(custom)
    manager.select(imported.qualified_id)
    assert manager.profile_notes("gemma-4.yaml") == "Custom Gemma explanation"
    assert manager.profile_notes("qwen3_8.yaml").startswith("Qwen3.8-27B")

    for bad in ({"../evil.yaml": "x"}, {"gemma-4.yaml": ""}, ["gemma-4.yaml"]):
        custom.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "id": "bad-notes",
                    "name": "Bad",
                    "locale": "xx-XX",
                    "strings": {},
                    "profile_notes": bad,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(LanguagePackError):
            manager.import_pack(custom, replace=True)


def _gui_help_strings() -> set[str]:
    """Every English help/tooltip/dialog string the GUI builds from constants.

    Mirrors the catalogue rule: two-level tooltips (``_setting_tooltip``),
    strings passed through ``_tr``/``translate``, the model-row tooltip
    constants, metric help, fork tooltip constants, and the performance
    target descriptions must all be translatable.
    """
    source = (ROOT / "qt_launcher.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found: set[str] = set()

    def is_const(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else (func.attr if isinstance(func, ast.Attribute) else None)
            )
            if name == "_setting_tooltip":
                found.update(a.value for a in node.args if is_const(a))
            elif name in ("_tr", "translate", "tr") and node.args and is_const(node.args[0]):
                found.add(node.args[0].value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
            if (
                target.startswith("MODEL_TOOLTIP_")
                or target in ("_FORK_TOOLTIP_SUMMARY", "_FORK_TOOLTIP_TECHNICAL")
            ) and is_const(node.value):
                found.add(node.value.value)
            if target == "_METRIC_HELP" and isinstance(node.value, ast.Dict):
                found.update(v.value for v in node.value.values if is_const(v))
    from performance_target import PERFORMANCE_TARGETS

    found.update(target.description for target in PERFORMANCE_TARGETS.values())
    found.update({TOOLTIP_SUMMARY_LABEL, TOOLTIP_TECHNICAL_LABEL})
    return {text for text in found if text.strip()}


def test_every_gui_help_string_is_translated_in_every_builtin_pack(tmp_path) -> None:
    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    assert not manager.errors
    required = _gui_help_strings()
    assert len(required) >= 250
    for pack in manager.available():
        missing = sorted(text for text in required if text not in pack.strings)
        assert missing == [], f"{pack.qualified_id} lacks {len(missing)} strings: {missing[:5]}"
        if pack.qualified_id != DEFAULT_LANGUAGE_ID:
            untranslated = [
                text
                for text in required
                if pack.strings[text] == text and len(text) > 40
            ]
            assert untranslated == [], f"{pack.qualified_id}: {untranslated[:3]}"


def test_two_level_tooltips_are_translated_as_plain_text(tmp_path) -> None:
    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    summary = "Changes AutoTuner's interface language immediately."
    technical = (
        "Built-in JSON packs fall back to English (UK) for unknown strings. "
        "Custom packs are validated and copied into ~/.autotuner/languages."
    )
    html = setting_tooltip_html(summary, technical)
    assert "<b>In short:</b>" in html and "&#x27;" in html  # escaped apostrophe

    manager.select("builtin:de-DE")
    translated = manager.translate_tooltip(html)
    assert translated.startswith("<html><body style='max-width:520px'><p><b>Kurz gesagt:</b> ")
    assert "Technische Details:" in translated
    assert "Oberflächensprache" in translated
    assert "Built-in JSON" not in translated
    # Translating the translated HTML again is stable (idempotent on non-English).
    assert manager.translate_tooltip(translated) == translated

    # Dynamic suffixes and runtime-composed lines survive with translated prefixes.
    dynamic = setting_tooltip_html(
        summary, technical + "\nActive build: b10797_vulkan\nResolved path: L:\\x"
    )
    out = manager.translate_tooltip(dynamic)
    assert "Aktiver Build: b10797_vulkan" in out
    assert "Aufgelöster Pfad: L:\\x" in out
    suffixed = html + "<br><br>AUTOTUNER_CONTROL_API_KEY currently overrides this value."
    assert manager.translate_tooltip(suffixed).endswith(
        "<br><br>AUTOTUNER_CONTROL_API_KEY currently overrides this value."
    )

    # Bullet lines built from the performance-target registry are translated too.
    from performance_target import PERFORMANCE_TARGETS

    bullet = f"• balanced: {PERFORMANCE_TARGETS['balanced'].description}"
    assert manager.translate_text(bullet).startswith("• balanced: Standard.")

    # English selection restores the exact original HTML.
    manager.select(DEFAULT_LANGUAGE_ID)
    assert manager.translate_tooltip(html) == html


def test_widget_tooltips_follow_the_language_live(tmp_path) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qt_widgets = pytest.importorskip("PyQt6.QtWidgets")
    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    manager = LanguageManager(BUILTIN_LANGUAGES, tmp_path / "languages")
    root = qt_widgets.QWidget()
    button = qt_widgets.QPushButton("Rename", root)
    button.setToolTip(
        setting_tooltip_html(
            "Rename the selected Custom profile for this model and performance mode.",
            "Rename the selected Custom profile for this model and performance mode.",
        )
    )
    plain = qt_widgets.QLabel("x", root)
    plain.setToolTip("Hover for how this metric is collected.")

    manager.select("builtin:ru-RU")
    manager.apply_to(root)
    assert "Кратко:" in button.toolTip()
    assert "Переименовать выбранный пользовательский профиль" in button.toolTip()
    assert plain.toolTip() == "Наведите курсор, чтобы узнать, как собирается эта метрика."

    manager.select("builtin:fr-FR")
    manager.apply_to(root)
    assert "En bref :" in button.toolTip()
    assert plain.toolTip() == "Survolez pour savoir comment cette métrique est collectée."
    root.close()
    assert app is qt_widgets.QApplication.instance()
