# How to Push This Submission to a Public GitHub Repository

The folder is already a git repo with one commit ready to push.
You just need to create a public repo on GitHub and run **two commands**.

---

## Step 1 — Create the repo on GitHub (browser)

1. Go to https://github.com/new
2. **Repository name:** `dual-arm-diffusion-ablation` (or anything you like)
3. **Visibility:** **Public** ✅
4. **Do NOT** initialize with README, .gitignore, or license
   (the local repo already has them)
5. Click **Create repository**
6. Copy the URL it shows you, e.g.
   `https://github.com/yourname/dual-arm-diffusion-ablation.git`

## Step 2 — Push from this folder

Open Git Bash / PowerShell in `Student4_Submission/` and run:

```bash
git remote add origin https://github.com/<YOUR_USERNAME>/dual-arm-diffusion-ablation.git
git push -u origin main
```

If GitHub asks for authentication, use a Personal Access Token (Settings →
Developer settings → Personal access tokens → Tokens (classic) → Generate)
as the password.

## Step 3 — Update the URL in two places

Replace `<YOUR_USERNAME>` with your actual GitHub username in:

1. `README.md` (top of the file)
2. `report/student4_report.tex` (Appendix A)

Then re-compile the LaTeX report so the printed PDF shows the real link.

---

## Optional — Install GitHub CLI for one-step pushes next time

```bash
# Windows (via winget)
winget install --id GitHub.cli

# After install:
gh auth login
gh repo create dual-arm-diffusion-ablation --public --source . --push
```
