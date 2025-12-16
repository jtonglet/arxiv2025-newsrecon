import os
from tqdm import tqdm
import time
from utils import *


def scrape_image(url,file_path, sleep=0.5):
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
        with Image.open(BytesIO(image_content)) as img:
            img.save(file_path + '.png')
            time.sleep(sleep)
    else:
        print(req.status_code)
        

if __name__ == "__main__":
    os.makedirs("data/TARA_dataset/non_gt_img_guardian", exist_ok=True)
    for file in os.listdir('data/processed_articles'):
        if 'guardian' in file:
            data = load_json('data/processed_articles/' + file)  

        for d in tqdm(range(len(data))):
            url = data[d]['image_url']
            if url!='':
                file_path = os.path.join("data/TARA_dataset/non_gt_img_guardian", '_'.join(url.split('/')[-2:]))
                if not os.path.exists(file_path):
                    #The image has already been stored in a prior run
                    if '68x68' not in url:
                        #The image is not a thumbnail
                        try:
                            scrape_image(url,file_path)
                            data[d]['image_path'] = '_'.join(url.split('/')[-2:]) + '.png'
                        except:
                            data[d]['image_path'] = ""
                else:
                    data[d]['image_path'] = '_'.join(url.split('/')[-2:]) + '.png'

        #Remove all articles from the corpus for which an image could not be downloaded
        data = [d for d in data if d['image_path']!='']
        with open(os.path.join('data/processed_articles', file), 'w') as f:
            json.dump(data, f, indent=4)
                    
    
    