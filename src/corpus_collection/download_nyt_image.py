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
            file_path = 'data/TARA_dataset/non_gt_img/' + file_path
            img.save(file_path + '.png')
            time.sleep(sleep)
    else:
        print(req.status_code)


if __name__ == "__main__":
    os.makedirs("data/TARA_dataset/non_gt_img",exist_ok=True)
    data = []
    for file in os.listdir('data/processed_articles'):
        data += load_json('data/processed_articles/' + file) 

    qwen_class = []
    for file in os.listdir('data/qwen2.5_article_class'):
        qwen_class += load_json(os.path.join('data/qwen2.5_article_class', file))
          
    for d in tqdm(range(len(data))):
        if 'nytimes.com' in data[d]['web_url']:
            #Only NY Times articles
            file_path = '_'.join(data[d]['web_url'].split('nytimes.com/')[1].split('.')[0].split('/'))
            if len(data[d]['multimedia']) > 0:
                #The article needs to have at least one image of sufficient size
                if data[d]['multimedia'][0]['height'] > 75 and data[d]['multimedia'][0]['width'] > 75:
                    #The article needs to be of category 1
                    if qwen_class[d]['output'][0].lower()=='category 1':
                        url = f"https://static01.nyt.com/{data[d]['multimedia'][0]['url']}"
                        try:
                            scrape_image(url,file_path)
                        except:
                            print('failed')
                            pass
    
    