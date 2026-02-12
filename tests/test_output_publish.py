from datetime import date
from pathlib import Path

from metatranscribe.output.publish import (
    build_publish_filename,
    copy_published_markdown,
    sanitize_suggested_name,
)


def test_sanitize_suggested_name_normalizes_text() -> None:
    assert sanitize_suggested_name("Weekly Sync: Roadmap + Risks") == "weekly_sync_roadmap_risks"
    assert sanitize_suggested_name("  ") == "untitled"
    assert sanitize_suggested_name("Caf\u00e9 Updates") == "cafe_updates"


def test_build_publish_filename_formats_date_and_name() -> None:
    actual = build_publish_filename(date(2026, 2, 12), "Weekly Sync")
    assert actual == "2026-02-12_weekly_sync.md"


def test_copy_published_markdown_appends_suffix_on_collision(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("# Title\n", encoding="utf-8")
    destination = tmp_path / "published"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "2026-02-12_weekly_sync.md").write_text("existing", encoding="utf-8")

    copied = copy_published_markdown(source, destination, "2026-02-12_weekly_sync.md")

    assert copied.name == "2026-02-12_weekly_sync_2.md"
    assert copied.read_text(encoding="utf-8") == "# Title\n"
