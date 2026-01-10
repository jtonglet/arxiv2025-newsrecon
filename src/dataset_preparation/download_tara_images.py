from tqdm import tqdm
import time
import os
import requests as rq
from io import BytesIO
from utils import *

def scrape_image(url,file_path):
    '''
    Scrape an image given its url and store it locally as a png file.
    '''
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        req = rq.get(url, stream=True, timeout=(10,10), headers=headers)
    except Exception as e:
        print(e)
        return None
    if req.status_code == 200 and 'image' in req.headers.get('Content-Type', ''):
        image_content = req.content
        image = Image.open(BytesIO(image_content))
        image.verify()
        with Image.open(BytesIO(image_content)) as img:
            file_path = file_path
            img.save(file_path + '.png')


#Download all images of the TARA dataset 
for split in ['train', 'dev', 'test', 'interest']:
    os.makedirs(f'TARA_dataset/img/{split}/', exist_ok=True)
    data_path = f'TARA_dataset/input/gold_{split}.jsonl' if split!='train' else f'TARA_dataset/input/{split}.jsonl'
    data = load_jsonl(data_path)
    for t in tqdm(range(len(data))):
        path= f'TARA_dataset/img/{split}/{t}'
        scrape_image(data[t]['image_url'], path)
        time.sleep(2)