# DEPLOYMENT.md — GitHub + Streamlit Community Cloud

This project cannot be auto-deployed from a chat session (that would require
access to your GitHub and Streamlit accounts). Instead, everything is wired so
the **one-time** connection takes about two minutes, and **every push after that
redeploys automatically**.

There are two one-time steps: (A) push to GitHub, (B) link the repo on Streamlit
Community Cloud. A GitHub Actions workflow then keeps `main` green on every push.

---

## A. Push to GitHub (one time)

### Option 1 — use the helper script
```bash
cd qsqfs-diabetes-fs
./push_to_github.sh https://github.com/<your-username>/<your-repo>.git
```
The script runs `git init`, commits everything, adds your remote, and pushes to
`main`.

### Option 2 — manual
```bash
cd qsqfs-diabetes-fs
git init -b main
git add .
git commit -m "QSQ-FS: multimodal diabetes feature selection + Streamlit app"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> Create the empty repository first at <https://github.com/new> (no README, no
> .gitignore — this project already has them). If you use HTTPS and have 2FA, you
> will be prompted for a **Personal Access Token** as the password
> (Settings → Developer settings → Personal access tokens → Fine-grained, with
> `Contents: Read and write` on the repo).

After pushing, open the **Actions** tab on GitHub: the CI workflow
(`.github/workflows/ci-deploy.yml`) runs lint + smoke tests on Python 3.11 and
3.12. A green check means the code that will deploy is sound.

---

## B. Link the repo on Streamlit Community Cloud (one time)

1. Go to <https://share.streamlit.io> and sign in with GitHub. Authorise access
   to the repository you just pushed.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `<your-username>/<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. (Optional) Click **Advanced settings** → set **Python version** to `3.11`.
5. Click **Deploy**. The first build installs `requirements.txt` (core only — no
   torch, so it is quick) and launches the dashboard.

That's it. From now on, **every `git push` to `main` triggers an automatic
redeploy** — no further action needed.

---

## After deployment — automatic redeploys

```bash
# make changes, then:
git add .
git commit -m "tweak: ..."
git push
```

Streamlit Community Cloud detects the push and rebuilds. The Actions workflow
runs in parallel to confirm the push is green.

---

## Configuration & secrets

- **No secrets are required** for demo mode.
- For real MIMIC-IV data you must **not** commit the dataset (it is credentialed).
  `.gitignore` already excludes `data/mimic-iv-demo/*`. Options:
  - Run locally / on a private server with the data mounted, **or**
  - Upload a CSV through the app's uploader, **or**
  - Use Streamlit **Secrets** (App → Settings → Secrets) for any credentials your
    own loader needs. Never paste PHI into a public app.

---

## Resource notes (Streamlit Community Cloud)

- The free tier has limited RAM/CPU. The default **sklearn** MLP keeps the image
  small and cold starts fast. Enabling `torch` in `requirements.txt` works but
  adds a large download and may approach memory limits — prefer torch for local
  runs.
- Keep nested-CV folds and iteration budgets modest in the deployed app (the
  sidebar sliders let users dial these up only when needed).
- If a build fails on a heavy optional dependency, confirm it is still commented
  out in `requirements.txt`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `git push` rejected (auth) | Use a Personal Access Token as the HTTPS password, or set up SSH keys. |
| Streamlit build times out | Ensure `torch`/`xgboost`/`shap` are commented in `requirements.txt`. |
| App boots but "No data" | Demo mode on? If using real data, check `paths.data_root` points at the folder containing `hosp/` and `icu/`. |
| Wrong Python version | Set it under Streamlit **Advanced settings**; `runtime.txt`/`.python-version` request 3.11. |
| CI red but app fine | Open the **Actions** log; usually a lint (`E9/F-series`) or import error flagged early — fix and push. |

---

## Local Docker (optional)

If you prefer a container instead of Streamlit Cloud:

```dockerfile
# Dockerfile (create at repo root if you want this)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t qsqfs . && docker run -p 8501:8501 qsqfs
```
