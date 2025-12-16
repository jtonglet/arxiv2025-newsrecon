from utils import *
import pandas as pd
import os
from tqdm import tqdm
import requests
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import spacy
import string

nlp = spacy.load("en_core_web_sm")

#Prepare API key
GUARDIAN_API_KEY = os.getenv("YOUR_GUARDIAN_API_KEY")


def query_guardian(query_words, 
                   date_str, 
                   time_delta=2 #time delta max 2 days after or before the defined string                
                  ):
    """
    Query the Guardian Content API with the specified query words and date interval.

    Parameters:
        query_words (list): List of words to search for.
        date_str (str): A date in "YYYY-MM-DD" format. The search will cover a period
                        from two days before to two days after this date.
    """
    base_url = "https://content.guardianapis.com/search"
    # Parse the date string
    try:
        query_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError("date_str must be in format YYYY-MM-DD")
    # Calculate the date interval
    if time_delta > 0:
        start_date = query_date - timedelta(days=time_delta)
        end_date = query_date + timedelta(days=time_delta)
    else:
        start_date, end_date = query_date, query_date
    if len(query_words) > 0:
        query_string = " OR ".join(query_words).translate(str.maketrans('', '', string.punctuation))
        query_string = query_string.replace('’', '').replace('“', '').replace('”', '')
    else:
        query_string = ''
    #setup session
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    if query_string!='':
        params = {
            "q": query_string,
            "from-date": start_date.isoformat(),
            "to-date": end_date.isoformat(),
            "section": "world|us-news|australia-news|uk-news|politics|society|environment",
            "api-key": GUARDIAN_API_KEY,
            "page-size": 50,  
            "lang":"en",
            "order-by": "relevance",
            "show-fields": "trailText,headline,body,score,thumbnail",
            "show-elements": "image",
            "show-tags": "keyword"
        }
    else:
        params = {
            "from-date": start_date.isoformat(),
            "to-date": end_date.isoformat(),
            "section": "world|us-news|australia-news|uk-news|politics|society|environment",
            "api-key": GUARDIAN_API_KEY,
            "page-size": 50,  
            "lang":"en",
            "order-by": "relevance",
            "show-fields": "trailText,headline,body,score,thumbnail",
            "show-elements": "image",
            "show-tags": "keyword"
        }
        
    
    response = session.get(base_url, params=params)
    
    # If the API returns a rate limit error, wait and retry once
    if response.status_code == 429:
        print("Rate limit reached. Waiting for 60 seconds before retrying...")
        time.sleep(60)
        response = session.get(base_url, params=params)
    
    response.raise_for_status()
    return response.json()['response']['results']


def extract_entities(text):
    """
    Extract named entities from a given text using spaCy's NER.
    """
    doc = nlp(text)
    valid_labels = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']
    return [ent.text.lower() for ent in doc.ents if ent.label_ in valid_labels]


def find_missing_dates(date_list):
    '''
    Find the dates between 2022 and 2023 that are not covered by the instance of the dataset
    '''
    # Convert input strings to datetime.date objects
    input_dates = set(datetime.strptime(d, "%Y-%m-%d").date() for d in date_list)
    # Define the start and end dates for the interval (2022-2023)
    start_date = datetime.strptime("2022-01-01", "%Y-%m-%d").date()
    end_date = datetime.strptime("2023-12-31", "%Y-%m-%d").date()
    # Generate the full set of dates in the interval
    total_days = (end_date - start_date).days + 1
    full_date_set = {start_date + timedelta(days=i) for i in range(total_days)}
    # Find the missing dates by set difference
    missing_dates = sorted(full_date_set - input_dates)
    # Format the missing dates back to string if needed
    missing_dates_str = [d.strftime("%Y-%m-%d") for d in missing_dates]
    return missing_dates_str


def find_url(data):
    '''
    Search for the image URL in a Guardian article, if available
    '''
    image_url = ''
    if 'elements' in data.keys():
        if len(data['elements']) > 0:
            if 'assets' in data['elements'][0].keys():
                if len(data['elements'][0]['assets']) > 0:
                    if 'typeData' in data['elements'][0]['assets'][0].keys():
                        if 'secureFile' in data['elements'][0]['assets'][0]['typeData'].keys():
                            image_url =  data['elements'][0]['assets'][0]['typeData']['secureFile']
    return image_url  
    

if __name__=='__main__':

    #Load data
    #TARA
    tara_data = []
    for file in os.listdir('data/tara_articles'):
        tara_data += load_json('data/tara_articles/' + file)
    tara_data = pd.DataFrame(tara_data).sort_values(by='pub_date')
    tara_data['idx'] = tara_data.index
    tara_data = tara_data.to_dict(orient='records')
    #5pils-OOC
    pils = load_json('data/5pils_ooc/test.json')[::2]
    for p in range(len(pils)):
        pils[p]['date'] = pils[p]['date_numeric_label'][0]
        pils = [p for p in pils if p['date']!='n']
        pils = [p for p in pils if int(p['date'][:4]) >= 2010]
        pils = pd.DataFrame(pils).sort_values(by='date')
        pils['idx'] = pils.index
        pils = pils.to_dict(orient='records')

    #Find dates in 2022-2023 not covered by 5Pils 
    #We should query guardian articles for all dates in 2022-2023, not only those covered by 5Pils-OOC instances
    covered_dates = [d['date'].split('T')[0] for d in pils]
    missing_dates = find_missing_dates(covered_dates)

    #queries for tara
    queries_list = []
    current_date = ''
    query_idx = ''
    query= []
    for d in tqdm(range(len(tara_data))):
        idx = tara_data[d]['idx']
        if tara_data[d]['pub_date'].split('T')[0]!= current_date:
            if query_idx!='':
                #save previous date data to dict
                queries_list.append({
                    'pub_date': current_date,
                    'gt_articles_idx': query_idx.split('-'),
                    'query': list(set(query))
                }
                )
            #Set a new date
            current_date = tara_data[d]['pub_date'].split('T')[0]
            query_idx = f"{idx}"   
            query = []
        else:
            query_idx += f"-{idx}"
            #Add NYT keywords
            nyt_keywords = [k['value'].lower() for k in tara_data[d]['keywords'] if k['name'] in ['glocations', 'persons', 'organizations']]
            query += nyt_keywords
            #Add abstract keywords
            abstract_keywords = extract_entities(tara_data[d]['abstract'])
            query += abstract_keywords

    #queries for 5pils_ooc
    current_date = pils[0]['date'].split('T')[0]
    idx = pils[0]['idx']
    query_idx = f"{idx}"
    query = []
    for d in tqdm(range(0, len(pils))):
        if pils[d]['date'].split('T')[0]!= current_date:
            #save previous date data to dict
            queries_list.append({
                'pub_date': current_date,
                'gt_articles_idx': query_idx.split('-'),
                'query': list(set(query))
            }
            )
            #Set a new date
            current_date =pils[d]['date'].split('T')[0]
            query_idx = f"{idx}"   
            query = []
        else:
            query_idx += f"-{idx}"
            query += extract_entities(pils[d]['true_caption'])

      
    #Download articles     
    queries_list_part1 = [d for d in queries_list if d['pub_date'].split('-')[0] not in ['2022', '2023']]
    queries_list_part2 =  [d for d in queries_list if d['pub_date'].split('-')[0] in ['2022', '2023']]

    limit_articles = 2
    counter =0
    for q_list in [queries_list_part1, queries_list_part2]:
        results = []
        for q in tqdm(range(len(q_list))):
            query = q_list[q]['query']
            date_str= q_list[q]['pub_date']
            try:
                search_results = query_guardian(query, date_str, 2)[:limit_articles]
                search_results = [r for r in search_results if r['imageUrl']!='']
                for r in range(len(search_results)):
                    search_results[r]['gt_articles_idx'] = q_list[q]['gt_articles_idx']
                    search_results[r]['query'] = query
                    search_results[r]['web_url'] = search_results[r]['webUrl']
                    search_results[r]['image_url'] = search_results[r]['imageUrl']
                    search_results[r]['pub_date'] = search_results[r]['webPublicationDate']
                results += search_results
            except Exception as e:
                pass
            time.sleep(2)
        if len(results)!=0:
            if counter==0:
                output_path = 'data/processed_articles/guardian.json'
                counter +=1
            else:
                output_path='data/processed_articles/guardian_part2.json'
            with open(output_path, 'w') as file:
                json.dump(results, file, indent=4)
                results = []


    #Add the missing dates for 2022-2023
    limit_articles = 20
    results = []
    for q in tqdm(range(len(missing_dates))):
        date_str= missing_dates[q]
        query = []
        search_results = query_guardian(query, date_str, 0)[:limit_articles]
        for r in range(len(search_results)):
            search_results[r]['gt_articles_idx'] = []  #No corresponding ground truth article
            search_results[r]['query'] = query
            search_results[r]['web_url'] = search_results[r]['webUrl']
            search_results[r]['image_url'] = search_results[r]['imageUrl']
            search_results[r]['pub_date'] = search_results[r]['webPublicationDate']
        results += search_results
        time.sleep(2)
    with open('data/processed_articles/guardian_part2.json', 'a') as file:
        json.dump(results, file, indent=4)
        results = []


    # Final filtering steps --> remove duplicate URLs and articles without images
    for file in os.listdir('data/processed_articles/'):
        if 'guardian' in file:
            guardian_data = load_json('data/processed_articles/' + file)

            filtered_data = []
            web_urls_set = set()
            for d in tqdm(range(len(guardian_data))):
                if guardian_data[d]['web_url'] not in web_urls_set:
                    #Remove duplicate URLs
                    new_entry = guardian_data[d]
                    #Remove body to save storage  space
                    soup = BeautifulSoup(new_entry['fields']['body'], 'html.parser')
                    p_tags = soup.find_all('p')
                    # Get only the first 5 <p> tags of the article's body
                    first_five = p_tags[:5]
                    new_entry['fields']['body'] = "\n".join(str(tag) for tag in first_five)
                    new_entry[d]['fields']['body'] = new_entry[d]['fields']['body'].replace('<p>','').replace('</p>', '')
                    new_entry['image_url']  = find_url(guardian_data[d])
                    if new_entry['image_url'] != '':
                        #Only include article if it has an image URL
                        new_entry['has_image'] = True

                        filtered_data.append(new_entry)
                    web_urls_set.add(guardian_data[d]['web_url'])
            
            with open('data/processed_articles/' + file, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=4, ensure_ascii=False)



        
