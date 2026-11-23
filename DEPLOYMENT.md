# Deployment Guide: CineMatch on Streamlit Community Cloud

This guide walks you through deploying **CineMatch** to the web for free using **Streamlit Community Cloud**, giving you a live public link (`https://<your-app-name>.streamlit.app`) that you can submit for college or share with anyone.

---

## Prerequisites

1. A free **GitHub account** ([github.com](https://github.com)).
2. Git installed on your computer (already installed).

---

## Step 1: Push Your Project to GitHub

### 1.1 Create a new repository on GitHub
1. Open your browser and go to [github.com/new](https://github.com/new).
2. Name the repository: `CineMatch` (or `Movie-Recommendation-System`).
3. Set the repository visibility to **Public**.
4. Leave "Add a README file", ".gitignore", and "license" **unchecked** (we already have them in our project).
5. Click **Create repository**.

### 1.2 Push the code from your PowerShell terminal
Open PowerShell and run:

```powershell
# Navigate into the project folder
cd "C:\Users\HP\Downloads\AATP\Movie-Recommendation-System"

# Rename current branch to main
git branch -M main

# Link to your new GitHub repository (replace with your actual GitHub username and repo name)
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/CineMatch.git

# Push the code to GitHub
git push -u origin main
```

*(If prompted, sign in to your GitHub account to authorize the push).*

---

## Step 2: Deploy on Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** (or [streamlit.io/cloud](https://streamlit.io/cloud)).
2. Click **"Sign in with GitHub"** and authorize Streamlit.
3. Once logged in, click the **"New app"** (or **"Create app"**) button.
4. Fill in the deployment form:
   - **Repository:** Select `YOUR_GITHUB_USERNAME/CineMatch`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (Optional) Choose a custom subdomain like `cinematch-movies`
5. Click **"Deploy!"**.

---

## Step 3: View Your Live Website

- Streamlit will now spin up a container, install `requirements.txt`, load the theme from `.streamlit/config.toml`, and run `app.py`.
- In about **1 to 2 minutes**, your app will be live at:
  ```
  https://<your-chosen-name>.streamlit.app
  ```
- Any time you push new code to your GitHub repo, Streamlit Cloud will **automatically update** the live website!

---

## Alternative: Deploying on Hugging Face Spaces

If you prefer Hugging Face:
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Select **Streamlit** as the Space SDK.
3. Clone the Space repo or push your files there, and it will deploy instantly.
