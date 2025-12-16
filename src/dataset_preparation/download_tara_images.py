from tqdm import tqdm
import time
import os
from utils import *

#Download all images of the TARA dataset 
for split in ['train', 'dev', 'test', 'interest']:
    os.makedirs(f'TARA_dataset/img/{split}/', exist_ok=True)
    data_path = f'TARA_dataset/input/gold_{split}.jsonl' if split!='train' else f'TARA_dataset/input/{split}.jsonl'
    data = load_jsonl(data_path)
    for t in tqdm(range(len(data))):
        path= f'TARA_dataset/img/{split}/{t}'
        scrape_image(data[t]['image_url'], path)
        time.sleep(2)