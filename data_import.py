import os
import requests
import pandas as pd
from tqdm import tqdm

def download_emotion_csv():
    url = "https://www.dropbox.com/scl/fi/tzw4p0xht67zp1vu4nfko/emotion.csv?rlkey=y9awrtw97zgsah642ga7hp6dp&st=ndqtf08f&dl=1"
    data_dir = "data"
    filename = "emotion.csv"
    filepath = os.path.join(data_dir, filename)

    # Create 'data/' directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the file if it doesn't exist
    if not os.path.exists(filepath):
        print(f"Downloading {filename} into {data_dir}/ ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(len(chunk))
        print(f"\nDownloaded {filename} successfully!")
    else:
        print(f"{filename} already exists in {data_dir}/.")

    return pd.read_csv(filepath)
