from datetime import date
from pathlib import Path

from metatranscribe.output.publish import (
    build_publish_filename,
    copy_published_markdown,
    infer_date_from_filename,
    sanitize_suggested_name,
)


def test_infer_date_from_filename_parses_recorder_format() -> None:
    today = date(2026, 5, 21)
    assert infer_date_from_filename("May 21 at 11-23 AM.m4a", today) == date(2026, 5, 21)
    assert infer_date_from_filename("January 3 at 9-00 PM.m4a", today) == date(2026, 1, 3)


def test_infer_date_from_filename_uses_previous_year_for_future_dates() -> None:
    today = date(2026, 5, 21)
    assert infer_date_from_filename("May 22 at 8-00 AM.m4a", today) == date(2025, 5, 22)
    assert infer_date_from_filename("December 31 at 11-59 PM.m4a", today) == date(2025, 12, 31)


def test_infer_date_from_filename_returns_none_for_non_matching() -> None:
    today = date(2026, 5, 21)
    assert infer_date_from_filename("recording.m4a", today) is None
    assert infer_date_from_filename("2026-05-21.m4a", today) is None


def test_infer_date_from_filename_handles_leap_day() -> None:
    # Feb 29 exists in 2024; today is 2026 so we'd need to go back to 2024 — beyond one year, returns None
    assert infer_date_from_filename("February 29 at 12-00 PM.m4a", date(2026, 5, 21)) is None
    # But if today is in 2024 and Feb 29 hasn't passed yet, it resolves to 2024
    assert infer_date_from_filename("February 29 at 12-00 PM.m4a", date(2024, 3, 1)) == date(2024, 2, 29)


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
