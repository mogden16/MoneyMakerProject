from pathlib import Path


def test_legacy_audit_doc_exists():
    assert Path("docs/LEGACY_REPO_AUDIT.md").exists()


def test_legacy_migration_plan_exists():
    assert Path("docs/LEGACY_MIGRATION_PLAN.md").exists()
