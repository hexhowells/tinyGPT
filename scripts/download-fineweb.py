from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceFW/fineweb",  # or "HuggingFaceFW/fineweb-edu"
    repo_type="dataset",
    allow_patterns="sample/10BT/*",
    local_dir="/media/datasets/fineweb_10BT",
)