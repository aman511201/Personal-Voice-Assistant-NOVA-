#!/usr/bin/env python3
"""
Automated Deployment Script for Hugging Face Spaces
Deploys Nova AI Voice Assistant to Hugging Face Spaces (Docker SDK).
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load local environment
load_dotenv()

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("❌ Error: 'huggingface_hub' package not found. Run 'pip install huggingface_hub'.")
    sys.exit(1)


def deploy_to_huggingface(token=None, space_name="nova-voice-assistant"):
    token = token or os.getenv("HF_TOKEN")
    
    if not token:
        print("\n" + "=" * 65)
        print("🤗 HUGGING FACE SPACES DEPLOYMENT FOR NOVA")
        print("=" * 65)
        print("To deploy automatically, you need a Hugging Face Access Token with 'Write' role.")
        print("Get your token at: https://huggingface.co/settings/tokens\n")
        token = input("Enter your Hugging Face Token (starts with 'hf_...'): ").strip()

    if not token:
        print("❌ No token provided. Aborting deployment.")
        return

    print("\n[1/5] Authenticating with Hugging Face...")
    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"✅ Authenticated as: @{username}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    repo_id = f"{username}/{space_name}"
    print(f"\n[2/5] Creating Docker Space: {repo_id} ...")
    try:
        space_url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False
        )
        print(f"✅ Space ready: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"❌ Failed to create Space: {e}")
        return

    # Add GEMINI_API_KEY as Space Secret automatically if available in .env
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("\n[3/5] Configuring GEMINI_API_KEY Space Secret...")
        try:
            api.add_space_secret(repo_id=repo_id, key="GEMINI_API_KEY", value=gemini_key)
            print("✅ GEMINI_API_KEY configured securely in Space Secrets.")
        except Exception as e:
            print(f"⚠️ Note: Could not set Space Secret automatically ({e}). You can set it manually in Space Settings -> Secrets.")
    else:
        print("\n[3/5] ⚠️ GEMINI_API_KEY not found in local .env. Remember to add it in Space Settings -> Secrets.")

    project_dir = Path(__file__).resolve().parent
    print(f"\n[4/5] Uploading project files from {project_dir.name} to Hugging Face...")
    ignore_list = [
        ".env",
        ".env.local",
        "venv/**",
        ".venv/**",
        "__pycache__/**",
        "*.pyc",
        "*.wav",
        "*.webm",
        "voice_input.wav",
        "recorded.wav",
        "memory/*.json",
        "memory/*.key",
        "memory/*.md",
        ".git/**",
        "scratch/**"
    ]

    try:
        api.upload_folder(
            folder_path=str(project_dir),
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=ignore_list,
            commit_message="Deploy Nova AI Voice Assistant to Hugging Face Spaces"
        )
        print("✅ Files uploaded successfully!")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return

    print("\n[5/5] 🚀 DEPLOYMENT INITIATED!")
    print("=" * 65)
    print(f"🌟 Your Space is now building and will be live at:")
    print(f"👉 https://huggingface.co/spaces/{repo_id}")
    print("=" * 65)
    print("Hugging Face is building your Docker image. In 1-2 minutes, Nova will be running live on HTTPS with full microphone and audio access!\n")


if __name__ == "__main__":
    cli_token = sys.argv[1] if len(sys.argv) > 1 else None
    cli_name = sys.argv[2] if len(sys.argv) > 2 else "nova-voice-assistant"
    deploy_to_huggingface(token=cli_token, space_name=cli_name)
