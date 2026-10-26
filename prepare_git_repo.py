#!/usr/bin/env python3
"""
Safely initializes the local Git repository in ~/gemini-argon-hackathon,
enforces .gitignore rules so .env (GEMINI_API_KEY) is never exposed,
and creates the initial commit ready for VS Code 'Publish to GitHub'.
"""

import subprocess
from pathlib import Path

REQUIRED_GITIGNORE_LINES = [
    ".env",
    ".env.*",
    "!.env.example",
    "gemini_hack_3_11/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.py[cod]",
    ".pytest_cache/",
    "agent_outputs/",
    "pipeline_output/",
    "my_study_outputs/",
    "my_adam_outputs/",
    "*.zip",
    ".DS_Store",
]


def run_cmd(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 and "nothing to commit" not in res.stdout:
        print(f"[Info] {' '.join(cmd)} -> {res.stderr.strip()}")
    return res.stdout.strip()


def main() -> None:
    root = Path.cwd()
    print(f"Preparing Git repository in: {root}")

    # 1. If create_github_repo.py exists, run it to ensure all project files are up to date
    builder = root / "create_github_repo.py"
    if builder.exists():
        subprocess.run(["python3", str(builder)], check=True)

    # 2. Enforce strict .gitignore
    gitignore_path = root / ".gitignore"
    existing = gitignore_path.read_text(encoding="utf-8").splitlines() if gitignore_path.exists() else []
    merged = list(dict.fromkeys(existing + REQUIRED_GITIGNORE_LINES))
    gitignore_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print("Verified .gitignore (blocks .env and virtual environments).")

    # 3. Initialize local Git repository on branch 'main'
    if not (root / ".git").exists():
        run_cmd(["git", "init", "-b", "main"])
        print("Initialized empty Git repository on branch 'main'.")
    else:
        run_cmd(["git", "branch", "-M", "main"])
        print("Existing Git repository detected; ensured branch is 'main'.")

    # 4. Safety guard: untrack .env if it was ever staged previously
    run_cmd(["git", "rm", "--cached", "--ignore-unmatch", ".env"])

    # 5. Stage all safe repository files
    files_to_stage = [
        ".gitignore",
        ".env.example",
        "README.md",
        "AGENT_PROMPT.md",
        "requirements.txt",
        "app.py",
        "run_agent.py",
        "create_github_repo.py",
        "prepare_git_repo.py",
        "clinical_aggregator",
        "tests",
    ]
    for item in files_to_stage:
        if (root / item).exists():
            run_cmd(["git", "add", item])

    # 6. Verify .env is NOT in staged changes
    staged = run_cmd(["git", "diff", "--cached", "--name-only"]).splitlines()
    if ".env" in staged:
        raise RuntimeError("SECURITY ABORT: .env was detected in staged files!")

    # 7. Create initial commit
    run_cmd([
        "git",
        "commit",
        "-m",
        "Initial release: Clinical Study Source Data Aggregator AI Agent & SDTM-to-ADaM Pipeline",
    ])

    print("\n=== LOCAL GIT REPOSITORY READY ===")
    print("Staged & committed files:")
    tracked = run_cmd(["git", "ls-files"])
    for line in tracked.splitlines():
        print(f"  ✓ {line}")


if __name__ == "__main__":
    main()

