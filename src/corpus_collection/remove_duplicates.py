import os
from PIL import Image
import imagehash
from tqdm import tqdm

# This script ensures that no articles in the corpus have an image that is identical to one of the images from the TARA dataset

query_folders = "data/TARA_dataset/img/"
corpus_folder_1 = "data/TARA_dataset/non_gt_img_guardian"
corpus_folder_2 = "data/TARA_dataset/non_gt_img"

# Build a set of perceptual hashes for images in the query folder
query_hashes = set()
for folder_name in os.listdir(query_folders):
    query_folder = os.path.join(query_folders, folder_name)
    for filename in tqdm(os.listdir(query_folder)):
        file_path = os.path.join(query_folder, filename)
        try:
            with Image.open(file_path) as img:
                # Compute the perceptual hash of the image
                hash_val = imagehash.phash(img)
                query_hashes.add(str(hash_val))
        except Exception as e:
            print(f"Could not process {file_path}: {e}")

# Go through images in the corpus folder and remove duplicates found in the query folder
for filename in tqdm(os.listdir(corpus_folder_1)):
    file_path = os.path.join(corpus_folder_1, filename)
    try:
        with Image.open(file_path) as img:
            hash_val = imagehash.phash(img)
            if str(hash_val) in query_hashes:
                os.remove(file_path)
            # pass
                print(f"Removed duplicate image: {file_path}")
    except Exception as e:
        print(f"Could not process {file_path}: {e}")
        os.remove(file_path)

for filename in tqdm(os.listdir(corpus_folder_2)):
    file_path = os.path.join(corpus_folder_2, filename)
    try:
        with Image.open(file_path) as img:
            hash_val = imagehash.phash(img)
            if str(hash_val) in query_hashes:
                os.remove(file_path)
                print(f"Removed duplicate image: {file_path}")
            pass
    except Exception as e:
        print(f"Could not process {file_path}: {e}")
        os.remove(file_path)
