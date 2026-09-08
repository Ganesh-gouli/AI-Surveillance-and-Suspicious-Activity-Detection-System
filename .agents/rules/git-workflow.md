# Git Commit & Development History Policy

## Objective
Maintain a clean, professional, and continuous Git commit history on GitHub that reflects real development progress for visitors, recruiters, and interviewers.

## Core Rules & Workflow

1. **Commit on Meaningful Progress**:
   - Whenever a meaningful feature is implemented, a bug is fixed, the UI is enhanced, functionality is modified, code is refactored, or documentation/tests are improved, create a Git commit.
   - Do NOT create fake, empty, artificial, or meaningless commits just to inflate the commit graph.

2. **Commit Message Standards**:
   - Use clear, conventional, professional commit messages describing exactly what changed:
     - `feat: add real-time surveillance detection`
     - `feat: implement suspicious activity detection`
     - `feat: add weapon detection`
     - `fix: improve person detection accuracy`
     - `fix: resolve video processing issue`
     - `ui: improve surveillance dashboard`
     - `refactor: optimize video processing pipeline`
     - `docs: update project documentation`
     - `test: add automated detection verification tests`

3. **Logical Separation**:
   - Keep commits focused and logically grouped.
   - If multiple unrelated changes are made, commit them as separate, distinct commits rather than combining them into one mega-commit.

4. **Safety & Verification First**:
   - Verify code correctness, build status, or tests before committing. Ensure the project remains working and functional.
   - Never break existing project functionality unnecessarily.

5. **Security & Clean Hygiene**:
   - Never stage or commit secrets, API keys, passwords, `.env` files, credentials, `node_modules`, virtual environments (`.venv`, `venv`), databases (`*.db`), model weight binaries, large video/image datasets, or temporary cache files.
   - Respect `.gitignore` at all times.

6. **History Preservation**:
   - Never rewrite, force push, squash, or delete existing commit history unless explicitly instructed by the user.

7. **Automatic Push Workflow**:
   - After completing each meaningful change and verifying it, execute:
     ```powershell
     git add <changed-files>  # or git add . respecting .gitignore
     git commit -m "<type>: <concise professional description>"
     git push origin main
     ```
   - Keep the local `main` branch synchronized with GitHub `origin/main` at all times.
