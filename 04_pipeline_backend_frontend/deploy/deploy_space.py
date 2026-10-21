"""Upload the app to a Hugging Face Space (Docker SDK).

    hf auth login            # once, in your own terminal (token needs 'write')
    python 04_pipeline_backend_frontend/deploy/deploy_space.py --space YOUR_HF_USERNAME/sfaca-detector

Stages only what the image needs (src, backend, configs, docs, adapters), so datasets and
training runs never leave this machine.
"""
import argparse
import os
import shutil
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # repository root
M4 = os.path.join(ROOT, "04_pipeline_backend_frontend")
ADAPTER_FILES = {"adapter_model.bin", "classifier_head.pt"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="owner/name of the Space")
    ap.add_argument("--adapters", default=os.path.join(ROOT, "results", "adapters"))
    args = ap.parse_args()

    stage = tempfile.mkdtemp(prefix="sfaca_space_")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    # Same layout as the repository, so the image runs the code unchanged.
    shutil.copy2(os.path.join(ROOT, "sfaca_paths.py"), stage)
    shutil.copytree(os.path.join(ROOT, "configs"), os.path.join(stage, "configs"), ignore=ignore)
    for member in sorted(d for d in os.listdir(ROOT) if d[:2].isdigit() and os.path.isdir(os.path.join(ROOT, d))):
        shutil.copytree(os.path.join(ROOT, member, "src"), os.path.join(stage, member, "src"), ignore=ignore)
    for sub in ("backend", "web"):
        shutil.copytree(os.path.join(M4, sub), os.path.join(stage, "04_pipeline_backend_frontend", sub), ignore=ignore)
    for cl in os.listdir(args.adapters):
        src = os.path.join(args.adapters, cl)
        if not os.path.isdir(src):
            continue
        os.makedirs(os.path.join(stage, "adapters", cl))
        for f in ADAPTER_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(stage, "adapters", cl, f))
    shutil.copy2(os.path.join(M4, "deploy", "Dockerfile"), os.path.join(stage, "Dockerfile"))
    shutil.copy2(os.path.join(M4, "deploy", "SPACE_README.md"), os.path.join(stage, "README.md"))

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=stage, repo_id=args.space, repo_type="space",
                      commit_message="Deploy SFA-CA detector")
    shutil.rmtree(stage, ignore_errors=True)
    name = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"Uploaded. The Space builds for ~10 minutes: https://huggingface.co/spaces/{args.space}")
    print(f"Live URL once running: https://{name}.hf.space")


if __name__ == "__main__":
    main()
