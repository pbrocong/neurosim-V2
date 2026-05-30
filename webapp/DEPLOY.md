# Deploying the web app to Hugging Face Spaces

The Gradio app (`webapp/`) depends on the simulator engine (`simulator/`).
A Hugging Face Space is a self-contained git repo, so the deployment bundle
must contain **both** the web code and the engine modules side by side.

`webapp/app.py` adds two locations to `sys.path` at startup:

1. `../simulator` — the layout in this repo, and
2. its own folder — so it also works when the engine `.py` files are copied
   **flat** next to `app.py` (the layout a Space ends up with).

That means either of two deployment styles works with **no code changes**.

---

## Option A — flatten into a deploy folder (simplest)

Assemble one flat folder and push it as the Space:

```bash
# from the repo root
rm -rf .hf_deploy && mkdir .hf_deploy

# web app (app.py, gradio_app/, requirements.txt, README.md)
cp -R webapp/. .hf_deploy/

# engine modules, copied flat next to app.py
cp simulator/*.py .hf_deploy/

# sample device characteristic so the app has a default to load
mkdir -p .hf_deploy/datasets/zno_encap_48h
cp -R datasets/zno_encap_48h/. .hf_deploy/datasets/zno_encap_48h/

# (optional) ASL CSVs — large; only if you want the ASL dataset available
# cp datasets/sign_mnist_*.csv .hf_deploy/datasets/
```

Then create the Space and push:

```bash
# one-time: install the HF CLI and log in
pip install huggingface_hub
huggingface-cli login

# create a Gradio Space (replace <user>/<name>)
huggingface-cli repo create <user>/neurosim-v2 --type space --space_sdk gradio

cd .hf_deploy
git init -b main
git remote add origin https://huggingface.co/spaces/<user>/neurosim-v2
git add . && git commit -m "Deploy NeuroSim V2 web app"
git push -u origin main
```

The Space root now has `app.py`, `requirements.txt`, `README.md` (with the
Gradio frontmatter), the `gradio_app/` package, and the engine `.py` files —
everything the app imports.

---

## Option B — push the whole repo as the Space

Point the Space at `webapp/app.py` via frontmatter and keep the repo layout.
Hugging Face reads `README.md` **and** `requirements.txt` from the Space root,
so you would need a root `README.md` with Gradio frontmatter and
`app_file: webapp/app.py`, plus a root `requirements.txt`. This repo keeps the
HF frontmatter in `webapp/README.md` instead, so Option A is the recommended
path unless you want to restructure for whole-repo deployment.

---

## CPU notes

Free Spaces are CPU-only and time out around 5 minutes per request. Keep
default runs small (SimpleNet / Simple_CNN / LeNet5, a few epochs). The UI
already warns when a heavy model (VGG / AlexNet / ResNet18) is selected.
