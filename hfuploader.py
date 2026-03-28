from huggingface_hub import upload_folder

upload_folder(
    folder_path=".",                          # your local gestify folder
    repo_id="Adityax-94/gestify",
    repo_type="space",
    ignore_patterns=["*.git*", "__pycache__"]
)