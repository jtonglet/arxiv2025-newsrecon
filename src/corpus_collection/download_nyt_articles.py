import os
import pandas as pd
import json
from datetime import datetime
from tqdm import tqdm
import time
from pynytimes import NYTAPI
from dateutil.parser import parse
from utils import *

def filter_data(data):
    '''
    Filter function that keeps only the articles that have the same section names, document types, 
    and type of material as the articles of the TARA dataset.
    '''
    section_names_in_tara = ['Business Day', 'World', 'New York', 'U.S.', 'Science', 'Education', 'Week in Review', 'Arts', 'Technology', 
                             'Multimedia/Photos', 'Automobiles', 'Sunday Review', 'Your Money', 'Crosswords & Games', 'Real Estate', 
                             'Theater', 'The Upshot', 'T Magazine', 'Times Insider', 'Universal', 'NYT Now', 'Style', 'Climate', 
                             'The Learning Network', 'Reader Center', 'Lens', 'Well', 'Magazine', 'Corrections', 'Podcasts', 'Briefing', 
                             'International Home', 'Special Series']
    type_of_material_in_tara  = ['News', 'News Analysis', 'Slideshow', 'Video', 'Brief', 'Question', 'Special Report', 'Summary', 'Web Log', 
                                 'List', 'Biography', 'Text', 'Op-Ed', 'briefing', 'Interview', 'Editorial', 'An Analysis', 'Letter']
    filtered_data = [d for d in data if d['document_type']!='audio']
    #Keep same sections and type of news as TARA
    filtered_data = [d for d in filtered_data if d['section_name'] in section_names_in_tara]
    filtered_data = [d for d in filtered_data if d['type_of_material'] in type_of_material_in_tara]
    return filtered_data

def is_valid_date(date_string):
    #Helper function to verify the validate of a date string
    try:
        parse(date_string)
        return True
    except ValueError:
        return False
    

if __name__=='__main__':

    nyt_api_key = os.getenv("YOUR_NYT_API_KEY")
    nyt = NYTAPI(nyt_api_key, parse_dates=True)

    os.makedirs("data/processed_articles",exist_ok=True)

    #Get data for every month for the period 2012-2023
    for y in range(2012,2024,1):
        data = []
        for m in tqdm(range(1,13,1)):
            data +=  filter_data(nyt.archive_metadata(
                date = datetime(y, m, 1)
                        ))   
            time.sleep(12)
            for d in range(len(data)):
                if type(data[d]['pub_date'])!=str:
                    data[d]['pub_date'] =  data[d]['pub_date'].strftime('%Y-%m-%dT%H:%M:%S%z')
        with open(f"data/processed_articles/nyt_articles_{y}.json","w") as f:
            print(len(data))
            json.dump(data, f, indent=4)


    tara_data = []
    for file in os.listdir('data/tara_articles'):
        tara_data += load_json(os.path.join('data/tara_articles', file))
    tara_web_url = set([d['web_url'] for d in tara_data])
    tara_abstracts = set([d['abstract'] for d in tara_data])


    total = 0
    for file in os.listdir('data/processed_articles'):
        print('--------------------------------------')
        new_data = []
        data = load_json(os.path.join('data/processed_articles', file))
        print(f"Original length {len(data)}")
        for d in tqdm(range(len(data))):
            abstract = data[d]['abstract'].lower()
            #Remove some articles based on keywords that clearly indicate that they are generic
            if 'what to watch on' not in abstract and 'what we\u2019re reading' not in abstract and 'to the editor' not in abstract and  'corrections appearing in print' not in abstract and 'a collection of links' not in abstract:     
                if 'political news from today' not in abstract and 'lottery numbers' not in abstract and 'notable properties that have been recently' not in abstract and 'this word has appeared in' not in abstract and 'see what you know about the news' not in abstract:
                    if 'in case you need some puzzle' not in abstract and "what you need to know" not in abstract and "this week\u2019s properties are" not in abstract and 'readers react' not in abstract and 'corrections appeared in print'  not in abstract:
                        if 'fashion week photo diary' not in abstract and 'notable quotes from business articles' not in abstract and 'street-style photos from our NYTimesFashion' not in abstract:
                            if len(data[d]['multimedia']) > 0: 
                                #Needs to have at least one image
                                if data[d]['multimedia'][0]['width'] > 75 and data[d]['multimedia'][0]['height'] > 75:
                                    if not is_valid_date(abstract):
                                        if data[d]['web_url'] not in tara_web_url:
                                            #Verify that the article is not part of the TARA corpus
                                            if abstract not in tara_abstracts:
                                                new_data.append(data[d]) 

                                        
                                        
        #Drop all duplicates in the filtered data                           
        new_data = pd.DataFrame(new_data).sort_values(by='pub_date').drop_duplicates(subset=['web_url'], keep='first').drop_duplicates(subset=['abstract'], keep='first').to_dict(orient='records')
        #OPTIONAL: it might happen that the API does not return the exact same list of articles
        with open("data/corpus_articles_urls.txt", "r", encoding="utf-8") as f:
            urls = set([line.strip() for line in f if line.strip()])
        new_data = [d for d in new_data if d['web_url'] in urls]
        
        with open(f"data/processed_articles/{file}", "w") as output:
            json.dump(new_data, output, indent=4)
        print(f"New length {len(new_data)}")
        total += len(new_data)
    print('--------------------------------------')
    print(f"Total size {total}")