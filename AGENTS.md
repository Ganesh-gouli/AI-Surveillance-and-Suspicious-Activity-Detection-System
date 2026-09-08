# Agent Development Guidelines

## Git Commit & Public Progress Policy

Every agent working on this repository MUST follow these rules:

1. **Meaningful Commits**: Whenever a meaningful feature is added, bug fixed, UI improved, functionality modified, code refactored, or documentation updated, create a professional Git commit.
2. **Standard Message Conventions**:
   - `feat: <description>`
   - `fix: <description>`
   - `ui: <description>`
   - `refactor: <description>`
   - `docs: <description>`
   - `test: <description>`
3. **Continuous GitHub Sync**: Push each meaningful commit to GitHub (`git push origin main`) so public development progress remains visible and continuous.
4. **No Artificial Commits**: No empty, trivial, fake, or spam commits. Only real development progress.
5. **No History Rewrites**: Do not squash, reset, or rewrite past commits.
6. **Logical Atomicity**: Group related changes together; do not bundle unrelated modifications into one monolithic commit.
7. **Pre-Commit Verification**: Run relevant tests/lint checks before committing to ensure the project remains stable.
8. **Security & Clean Hygiene**: Never commit secrets, credentials, `.env`, `node_modules`, model weights, databases, large datasets, or cache files.
9. **Execution**:
   ```powershell
   git add .
   git commit -m "<appropriate message>"
   git push origin main
   ```
