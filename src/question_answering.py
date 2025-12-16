from tqdm import tqdm
from prompts import *
from utils import *
from llm_inference import *
from loaders import *
import transformers
import argparse
import os

transformers.set_seed(42)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tara', choices=['tara', '5pils_ooc'], help='Choose dataset. tara or 5pils_ooc')
    parser.add_argument('--questions', type=str, default="time-location")
    parser.add_argument('--num_instances', type=int, default=2000, help='number of query instances to run')
    parser.add_argument('--model', type=str, default='internvl3/8B/',  help="Name of the model to run")
    parser.add_argument('--evidence_file', type=str, default='',  help="Name of the folders containing the evidence to combine (- separated)")
    parser.add_argument('--num_evidence', type=int, default=3, help='Top k articles to provide from each list')
    parser.add_argument('--max_tokens', type=int, default=256,  help="Max number of generated tokens")
    parser.add_argument('--prompt', type=str, default='base', choices=['base', 'detective', 'celebrity'])
    parser.add_argument('--celebs_json', type=str, default='', help='Optional path to Rekognition results json. If empty, will pick a dataset default.')
    args = parser.parse_args()


    if args.dataset=='tara':
        data = load_jsonl('data/TARA_dataset/input/gold_test.jsonl')[:args.num_instances]
        ground_truth_web_urls = set([d['web_url'] for d in data])
    else:
        data = load_json('data/5pils_ooc/test.json')[::2][:args.num_instances]
        ground_truth_web_urls = set()
    template = args.model
    max_tokens = args.max_tokens
    questions = args.questions.split('-')

    # Load celebrity detections if the celebrity prompt is used
    celebs_by_image = {}
    if args.prompt == 'celebrity':
        celebs_path = 'rekog_celeb_tara_test.json' if args.dataset == 'tara' else 'rekog_celeb_5pils_ooc.json'
        if os.path.exists(celebs_path):
            celebs_raw = load_json(celebs_path)
            for k, v in celebs_raw.items():
                if isinstance(v, list):
                    names = [c.get('Name') for c in v if isinstance(c, dict) and c.get('Name')]
                    seen = set(); ordered = []
                    for n in names:
                        if n not in seen:
                            seen.add(n); ordered.append(n)
                    celebs_by_image[k] = ", ".join(ordered)
                else:
                    celebs_by_image[k] = "" 
        else:
            celebs_by_image = {}

    

    #Load the articles
    ref_articles = []
    article_paths = 'data/processed_articles'
    for file in os.listdir(article_paths):
        ref_articles += load_json(os.path.join(article_paths,file))
    for file in ['data/tara_articles/train.json']:
        ref_articles += load_json(file)
    web_urls_to_text_articles = {ref_articles[i]['web_url']:i for i in range(len(ref_articles))}


    #Load multimodal retrieval results      
    dataset_path = f"tara_test" if args.dataset=='tara' else "5pils_ooc_test"
    image_root = "data/TARA_dataset/img/" if args.dataset =='tara' else "data/5pils_ooc"
    articles = []
    if args.evidence_file!= '':
        retrieval_results = load_json(f'retrieval_results/{args.evidence_file}')
        for idx in range(len(data)):
            for r in range(args.num_evidence):
                if args.dataset=='tara':
                    web_url = retrieval_results[os.path.join(image_root, f"test/{idx}.png")][r]
                else:
                    web_url = retrieval_results[os.path.join(image_root, data[idx]['image_path'])][r]
                matching_article = ref_articles[web_urls_to_text_articles[web_url]]
                if 'abstract' in matching_article.keys():
                    #NYT articles
                    abstract = matching_article['abstract']
                    lead_paragraph = matching_article['lead_paragraph']
                else:
                    #guardian article
                    abstract = matching_article['fields']['headline'] + '\n' + matching_article['fields']['trailText']
                    #Max 500 characters from the body followed by trailing points
                    lead_paragraph = '.'.join(matching_article['fields']['body'][:500].split('.')[:-1])
                    if len(lead_paragraph) != 0:
                        lead_paragraph += '...'
                date =  matching_article['pub_date']
                location_keywords = ' - '.join([k['value'] for k in matching_article['keywords'] if k['name']=='glocations']).lower()
                image_path = os.path.join(image_root, f"test/{idx}.png") if args.dataset=='tara' else os.path.join(image_root, data[idx]['image_path'])
                articles.append({
                    'image_path': image_path,
                    'web_url': web_url,
                    'date': date,
                    'location': location_keywords,
                    'abstract': abstract,
                    'lead_paragraph': lead_paragraph

                })


    if template in ['GPT4V', 'GPT4o', 'gemini-1.5-flash', 'gemini-1.5-pro']:
        model, tokenizer = template, ''
    else:
        model, tokenizer = load_model(template, use_vllm=0) 
    #Main loop through the models
    pred_locations = {}
    for question in questions:
        model_answers = []
        for d in tqdm(range(len(data))):

            if args.dataset=='tara':
                im_path = f'data/TARA_dataset/img/test/{d}.png'
            else:
                im_path = f"data/5pils_ooc/{data[d]['image_path']}"
            if question=='time':
                if args.dataset=='tara':
                    correct_answer = data[d]['gold_time'] 
                else: 
                    correct_answer = data[d]['date_numeric_label']
                    correct_answer = correct_answer[0].split('T')[0] if correct_answer!='not enough information' else None
                correct_answer_set = get_time_hierarchy(correct_answer) if correct_answer else None
            else: 
                correct_answer = data[d]['gold_location'] if args.dataset=='tara' else data[d]['location']
                correct_answer = None if correct_answer=='not enough information' else correct_answer
                correct_answer_set = correct_answer.split(', ') if correct_answer else None
            if correct_answer: 
                #skip instances without correct answer
                if args.evidence_file!='':
                        arts = [ e for e in articles if e['image_path']==im_path] if articles!=[] else []
                    
                else:
                    arts = []

                celeb_key = im_path
                celeb_names = celebs_by_image.get(celeb_key, "") if args.prompt == 'celebrity' else ""
                prompt = create_prompt_qa(args.dataset, args.prompt, question, evidence=arts, celebrities=celeb_names)

                if d==0:
                    print(prompt)
                with torch.no_grad():
                    predicted_answer, cost = generate_answer(im_path, prompt, tokenizer, model, template, max_tokens)  
                    if predicted_answer!='':
                        if question=='time':
                            predicted_answer_set = get_time_hierarchy(predicted_answer)
                        else:
                            predicted_answer_set = predicted_answer.lstrip().split(',')
                            pred_locations[im_path] = predicted_answer
                    else:
                        predicted_answer_set = []        
                    model_answers.append({'image_path':im_path, 
                                        'predicted_answer':predicted_answer,
                                        'predicted_answer_set': predicted_answer_set,
                                        'correct_answer': correct_answer,
                                        'correct_answer_set': correct_answer_set,
                                        'cost': cost
                                        })


        os.makedirs('results', exist_ok=True)
        os.makedirs(f'results/{"-".join(template.split("/"))}/', exist_ok=True)

        if args.evidence_file != '':
            subdir = f'{args.dataset}_test_with_evidence_{args.num_evidence}'
        else:
            subdir = f'{args.dataset}_test_{args.prompt}'

        os.makedirs(f'results/{"-".join(template.split("/"))}/{subdir}/', exist_ok=True)

        if args.evidence_file != '':
            output_path = f'results/{"-".join(template.split("/"))}/{subdir}/{args.dataset}_test_predictions_{question}.json'
        else:
            output_path = f'results/{"-".join(template.split("/"))}/{subdir}/{args.dataset}_test_predictions_{question}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(model_answers, f, indent=4)








