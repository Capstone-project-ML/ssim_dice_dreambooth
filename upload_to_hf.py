import os
from huggingface_hub import HfApi, login
from getpass import getpass

# --- Configuration ---

# Your Hugging Face username
HF_USERNAME = "your-hf-username" 

# The desired name for your new model repository
REPO_NAME = "my-awesome-multi-folder-model"

# A list of local folders you want to upload the contents of
LOCAL_FOLDERS_TO_UPLOAD = [
    "./model_files",
    "./tokenizer_files",
    "./other_assets"
]

# --- Main Script Logic ---

def main():
    """
    Main function to create a repo and upload multiple folders.
    """
    print("--- Hugging Face Multi-Folder Uploader ---")

    # 1. Authenticate with Hugging Face
    # It's recommended to use an environment variable for the token.
    # If not found, it will prompt for the token securely.
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Hugging Face token not found in environment variables.")
        hf_token = getpass("Please enter your Hugging Face write token: ")
    
    try:
        login(token=hf_token)
        print("✓ Successfully logged in to Hugging Face.")
    except Exception as e:
        print(f"✗ Failed to log in: {e}")
        return

    # Instantiate the HfApi client
    api = HfApi()

    # Construct the repository ID
    repo_id = f"{HF_USERNAME}/{REPO_NAME}"

    # 2. Create the repository on the Hub
    try:
        print(f"\nCreating repository '{repo_id}' on the Hub...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",  # Can be "dataset" or "space" as well
            exist_ok=True,      # If the repo already exists, don't raise an error
        )
        print(f"✓ Repository created or already exists: {repo_url}")
    except Exception as e:
        print(f"✗ Failed to create repository: {e}")
        return

    # 3. Upload each folder's contents
    print("\nStarting folder uploads...")
    for folder_path in LOCAL_FOLDERS_TO_UPLOAD:
        print(f"\n--- Uploading contents of '{folder_path}' ---")
        
        # Check if the local folder exists
        if not os.path.isdir(folder_path):
            print(f"✗ Warning: Folder '{folder_path}' not found. Skipping.")
            continue

        try:
            # The `upload_folder` function will upload all files in the folder.
            # It will preserve the directory structure inside the repo if you 
            # specify a `path_in_repo`. Here, we upload to the root.
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload content from {os.path.basename(folder_path)}"
            )
            print(f"✓ Successfully uploaded contents of '{folder_path}' to '{repo_id}'.")
        except Exception as e:
            print(f"✗ Failed to upload '{folder_path}': {e}")
            
    print("\n--- All operations complete! ---")
    print(f"Check your repository online at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()



