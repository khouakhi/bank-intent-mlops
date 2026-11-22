"""Copy the app and the pinned model version to a Hugging Face Space."""
import json
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
space = json.loads((ROOT / "deploy_config.json").read_text())["space_repo"]
model_cfg = json.loads((ROOT / "model_config.json").read_text())

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    shutil.copytree(ROOT / "app", tmp, dirs_exist_ok=True)
    shutil.copy(ROOT / "model_config.json", Path(tmp) / "model_config.json")
    shutil.copy(ROOT / "src" / "predict.py", Path(tmp) / "predict.py")
    api.upload_folder(folder_path=tmp, repo_id=space, repo_type="space",
                      commit_message=f"Deploy {model_cfg['model_repo']}@{model_cfg['revision']}")

print(f"Deployed {model_cfg['revision']} to https://huggingface.co/spaces/{space}")
