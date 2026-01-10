from datetime import datetime, timedelta, timezone
from utils import *

def parse_iso8601(dt_str):
    """
    Parse an ISO-8601 style timestamp tolerantly:
    """
    s = dt_str
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    else:
        # insert colon into timezone offset if it's in "+HHMM" or "-HHMM" form
        s = re.sub(r'([+-]\d{2})(\d{2})$', r'\1:\2', s)
    dt = datetime.fromisoformat(s)  # this yields an aware datetime if offset present
    return dt.astimezone(timezone.utc)

def dates_in_range(dt_str, days=5):
    #Verify that an article date falls within a given interval around the ground truth date
    dt = parse_iso8601(dt_str)
    result = []
    for offset in range(-days, days + 1):
        d = dt + timedelta(days=offset)
        result.append(d.strftime("%Y-%m-%dT%H:%M:%SZ").split('T')[0])
    return result

N_WINDOW = 7

if __name__=='__main__':

    #Load data
    tara_train = load_jsonl('TARA_dataset/input/train.jsonl')
    tara_dev = load_jsonl('TARA_dataset/input/gold_dev.jsonl')

    #Load corpus
    articles = []
    for file in os.listdir('processed_articles'):
            articles += load_json('processed_articles/' + file) 
    qwen_class = []
    for file in os.listdir('qwen2.5_article_class'):
        qwen_class += load_json('qwen2.5_article_class/' + file) 
    articles = [articles[g] for g in range(len(articles)) if qwen_class[g]['output'][0]=='Category 1']
    articles += load_json('gt_articles/train.json')
    articles = [articles[g] for g in range(len(articles)) if 'glocations' in [kw['name'] for kw in articles[g]['keywords']]]


    #First identify relevant articles for the TARA train set
    query_image_to_relevant_articles = {}
    for t in tqdm(tara_train):
        query_image_to_relevant_articles[t['web_url']] = {
            'location': [], 
            'time': []
            }

        gold_loc = t['locations'][0].split('(')[0].lower().strip() #the lowest level possible of the location ground truth
        for a in range(len(articles)-len(tara_train)): #Omit the TARA train articles for the train set
            if articles[a]['web_url']!=t['web_url']:
                loc_set = set([v.strip().lower().replace(')', '')   for k in articles[a]['keywords'] for v in k['value'].split('(') if k['name']=='glocations'])
                if gold_loc in loc_set:
                    query_image_to_relevant_articles[t['web_url']]['location'].append(articles[a]['web_url'])
                    if t['date'] is not None:
                        #If the location matches the ground truth AND there is a ground truth date available, search for event-relevant articles
                        pub_date = t['pub_date'].split('T')[0]
                        gold_date = t['date']
                        len_gold_date = len(gold_date.split('-'))  #3, 2, or 1
                        if 's' in gold_date: #Ground truth is a decade
                            continue
                        if len_gold_date==3:
                            #use YYYY-MM-DD gold_date directly:
                            if gold_date in  [d.split('T')[0] for d in dates_in_range(articles[a]['pub_date'], N_WINDOW)]:
                                #The article's date falls within the range
                                query_image_to_relevant_articles[t['web_url']]['time'].append(articles[a]['web_url'])
                        if len_gold_date==2:
                            if gold_date in pub_date:
                                #use pub_date as a proxy if it is more specific than gold_date
                                if pub_date in  [d.split('T')[0] for d in dates_in_range(articles[a]['pub_date'], N_WINDOW)]:
                                    query_image_to_relevant_articles[t['web_url']]['time'].append(articles[a]['web_url'])
                        if len_gold_date==1:
                            #Ground truth is at the year level
                            if gold_date in pub_date:
                                #use pub_date as a proxy if it is more specific than gold_date
                                if pub_date in  [d.split('T')[0] for d in dates_in_range(articles[a]['pub_date'], N_WINDOW)]:
                                    query_image_to_relevant_articles[t['web_url']]['time'].append(articles[a]['web_url'])
    #Save results
    with open('data/relevant_articles_sets/relevant_articles_tara_train.json', 'w') as f:
        json.dump(query_image_to_relevant_articles, f, indent=4)

    #TARA dev set
    query_image_to_relevant_articles = {}
    for t in tqdm(tara_dev):
        query_image_to_relevant_articles[t['web_url']] = {
            'location': [], 
            'time': []
            }

        gold_loc = t['gold_location'][0].split('(')[0].lower().strip() #the lowest level possible of the location ground truth
        for a in range(len(articles)):
            if articles[a]['web_url']!=t['web_url']:
                loc_set = set([v.strip().lower().replace(')', '')   for k in articles[a]['keywords'] for v in k['value'].split('(') if k['name']=='glocations'])
                if gold_loc in loc_set:
                    query_image_to_relevant_articles[t['web_url']]['location'].append(articles[a]['web_url'])
                    if t['gold_time'] is not None:
                        #If the location matches the ground truth AND there is a ground truth date available, search for event-relevant articles
                        pub_date = t['pub_date'].split('T')[0]
                        gold_date = t['gold_time']
                        len_gold_date = len(gold_date.split('-'))  #3, 2, or 1
                        if 's' in gold_date: #Ground truth is a decade
                            continue
                        if len_gold_date==3:
                            #use YYYY-MM-DD gold_date directly:
                            if gold_date in  [d.split('T')[0] for d in dates_in_range(articles[a]['pub_date'], N_WINDOW)]:
                                #The article's date falls within the range
                                query_image_to_relevant_articles[t['web_url']]['time'].append(articles[a]['web_url'])
                        if len_gold_date==2:
                            if gold_date in pub_date:
                                #use pub_date as a proxy if it is more specific than gold_date
                                if pub_date in  [d.split('T')[0] for d in dates_in_range(articles[a]['pub_date'], N_WINDOW)]:
                                    query_image_to_relevant_articles[t['web_url']]['time'].append(articles[a]['web_url'])
                        if len_gold_date==1:
                            #Ground truth is at the year level
                            if gold_date in pub_date:
                                #use pub_date as a proxy if it is more specific than gold_date
                                if pub_date in  [d.split('T')[0] for d in dates_in_range(articles[a]['pub_date'], N_WINDOW)]:
                                    query_image_to_relevant_articles[t['web_url']]['time'].append(articles[a]['web_url'])
    #Save results
    with open('data/relevant_articles_sets/relevant_articles_tara_dev.json', 'w') as f:
        json.dump(query_image_to_relevant_articles, f, indent=4)