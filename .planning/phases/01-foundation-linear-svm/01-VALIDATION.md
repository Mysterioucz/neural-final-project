---
phase: 1
slug: foundation-linear-svm
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-14
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest tests/unit` |
| **Full suite command** | `uv run pytest` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest tests/unit`
- **After every plan wave:** Run `uv run pytest`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | DATA-01 | unit | `ls data/data.csv` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | DATA-02 | unit | `uv run pytest tests/unit/test_preprocessing.py` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 2 | SVM-01 | unit | `uv run pytest tests/unit/test_svm_logic.py` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 2 | SVM-02 | unit | `uv run pytest tests/unit/test_svm_logic.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/unit/test_preprocessing.py` — stubs for DATA-02, DATA-04
- [ ] `tests/unit/test_svm_logic.py` — stubs for SVM-01, SVM-02, SVM-03, SVM-04
- [ ] `tests/conftest.py` — shared fixtures for toy datasets
- [ ] `pip install pytest pytest-mock` — ensure testing tools are available

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Final Evaluation Report | EVAL-03 | Visual Audit | Open accuracy report and verify it compares all kernels. |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 15s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending 2026-05-14
