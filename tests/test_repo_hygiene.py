"""Repository hygiene tests.

These guard the publication policy encoded in `.gitignore`: model weights, audio,
experiment artifacts and secrets must stay untracked, while the `.gitkeep`
placeholders that keep their directories in Git must remain tracked. Ignore rules
are easy to break by accident, and a broken rule can silently commit a model
checkpoint or a private recording.

They are skipped when the tests run outside a Git working tree (for example from
an exported archive), because `git check-ignore` needs a repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PLACEHOLDERS = (
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/metadata/.gitkeep",
    "experiments/.gitkeep",
    "models/.gitkeep",
)

MUST_BE_IGNORED = (
    # model weights and caches
    "models/model.nemo",
    "models/indicconformer.safetensors",
    "models/huggingface/hub/models--ai4bharat--x/snapshots/abc/model.bin",
    "models/checkpoints/epoch.ckpt",
    "weights.pt",
    # audio
    "data/raw/as_000001.wav",
    "data/processed/as_000001.wav",
    # experiment output
    "experiments/baseline_20260919_001/metrics.json",
    "experiments/baseline_20260919_001/predictions.jsonl",
    # secrets and local config
    ".env",
    ".env.local",
    "configs/my.local.yaml",
    # caches and clutter
    ".venv/lib/python3.11/site-packages/x.py",
    "src/assamese_asr/__pycache__/x.cpython-311.pyc",
    ".pytest_cache/v/cache/nodeids",
    ".ruff_cache/0.14/index",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(autouse=True)
def require_git_working_tree() -> None:
    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a Git working tree; ignore rules cannot be checked")


@pytest.mark.parametrize("relative_path", PLACEHOLDERS)
def test_placeholder_file_exists(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).is_file(), (
        f"{relative_path} keeps an otherwise-empty directory in Git; do not delete it"
    )


@pytest.mark.parametrize("relative_path", PLACEHOLDERS)
def test_placeholder_is_tracked(relative_path: str) -> None:
    completed = _git("check-ignore", relative_path)

    assert completed.returncode == 1, (
        f"{relative_path} is ignored by Git but must stay tracked "
        f"(check-ignore said: {completed.stdout.strip()})"
    )


@pytest.mark.parametrize("relative_path", MUST_BE_IGNORED)
def test_path_is_ignored(relative_path: str) -> None:
    completed = _git("check-ignore", "--no-index", relative_path)

    assert completed.returncode == 0, (
        f"{relative_path} would be committed: .gitignore must cover it"
    )


def test_gitignore_keeps_the_model_placeholder_rule() -> None:
    rules = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "models/*" in rules
    assert "!models/.gitkeep" in rules


def test_no_tracked_file_has_a_model_weight_extension() -> None:
    tracked = _git("ls-files").stdout.splitlines()
    forbidden = (".nemo", ".ckpt", ".pt", ".pth", ".safetensors", ".onnx", ".bin")

    offenders = [name for name in tracked if name.endswith(forbidden)]

    assert offenders == [], f"model weights are tracked in Git: {offenders}"


def test_no_tracked_audio_file() -> None:
    tracked = _git("ls-files").stdout.splitlines()
    forbidden = (".wav", ".mp3", ".flac", ".ogg", ".oga", ".opus", ".m4a")

    offenders = [name for name in tracked if name.endswith(forbidden)]

    assert offenders == [], f"audio files are tracked in Git: {offenders}"
