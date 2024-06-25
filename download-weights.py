import os
import requests
import shutil

def download_file(url, folder_path, filename, auth=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path):
        print(f"File already exists: {file_path}")
    else:
        headers = {}
        if auth:
            headers['Authorization'] = f'token {auth}'  # Hier Token einfügen, wenn nötig

        try:
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print(f"File successfully downloaded and saved: {file_path}")
            else:
                print(f"Error downloading the file. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading the file: {e}")

# Download models
download_file(
    "https://civitai.com/api/download/models/266360?type=Model&format=SafeTensor&size=pruned&fp=fp16",
    "models/stable_diffusion",
    "flat2DAnimerge_v45Sharp.safetensors"
)

download_file(
    "https://huggingface.co/philz1337x/rv60b1/resolve/main/realisticVisionV60B1_v60B1VAE.safetensors?download=true",
    "models/stable_diffusion",
    "realisticVisionV60B1_v60B1VAE.safetensors"
)

download_file(
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
    "models/AnimateDiff",
    "mm_sd_v15_v2.ckpt"
)

download_file(
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "models/ControlNet",
    "control_v11p_sd15_lineart.pth"
)

download_file(
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
    "models/ControlNet",
    "control_v11f1e_sd15_tile.pth"
)

download_file(
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth",
    "models/Annotators",
    "sk_model.pth"
)

download_file(
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth",
    "models/Annotators",
    "sk_model2.pth"
)

download_file(
    "https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16",
    "models/textual_inversion",
    "verybadimagenegative_v1.3.pt"
)

download_file(
    "https://huggingface.co/philz1337x/test/resolve/main/flownet.pkl?download=true",
    "models/RIFE",
    "flownet.pkl"
)

