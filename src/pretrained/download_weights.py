import os
import urllib.request

def download_swin_unetr_weights(target_dir="models"):
    os.makedirs(target_dir, exist_ok=True)
    url = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt"
    filepath = os.path.join(target_dir, "model_swinvit.pt")
    
    if not os.path.exists(filepath):
        print(f"Downloading Swin UNETR weights from {url}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Successfully downloaded to {filepath}")
        except Exception as e:
            print(f"Failed to download weights: {e}")
            print("Please download manually and place in the 'models' directory.")
    else:
        print(f"Weights already exist at {filepath}")

if __name__ == "__main__":
    download_swin_unetr_weights()
