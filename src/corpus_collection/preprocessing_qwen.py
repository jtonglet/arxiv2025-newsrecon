import argparse
import os
from tqdm import tqdm
from utils import *
from llm_inference import *
from loaders import *
from prompts import create_prompt_article_classification
import random


SEED = 42
random.seed(SEED)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='data/processed_articles', help="Folder containing the article files")
    parser.add_argument('--model', type=str, default='qwen2.5/7B/',  help="Name of the model to run")
    parser.add_argument('--use_vllm', type=int, default = 1, help="set to 1 to use vllm for inference")
    parser.add_argument('--batch_size', type=int, default = 64, help="Number of prompts to instances to provide to the llm in one batch")
    parser.add_argument('--max_tokens', type=int, default=20,  help="Max number of generated tokens")
    
    args = parser.parse_args()

    #Prepare output dir
    root = f"{args.model.split('/')[0]}_article_class"
    os.makedirs(root, exist_ok=True)

    #Load the model
    template = args.model
    model, tokenizer = load_model(template, use_vllm=args.use_vllm) 

    #Load the articles
    for file in os.listdir(args.input_folder):
        data = load_json(os.path.join(args.input_folder, file))
        #iterate through the dataset
        if os.path.exists(os.path.join(root,f"{file}")):
            results = load_json(os.path.join(root,f"{file}"))
            already_done = set([r['web_url'] for r in results])
            print(f"Loading {len(already_done)} already captioned instances")
        else:
            results = []
            already_done = set()

        print('--------------------------------')
        print(f'Before removing already done {len(data)}')
        data = [data[d] for d in range(len(data)) if data[d]['web_url'] not in already_done]
        print(f'After removing already done {len(data)}')
        print('--------------------------------')
        
        batch_count = len(results)
        num_batches = (len(data) - len(results)+ args.batch_size - 1) // args.batch_size
        for iter in tqdm(range(num_batches)):
            prompts = []
            web_urls = []
            images = []
            input_texts = []
            for b in range(batch_count, min(batch_count+args.batch_size, len(data)),1):
                web_url = data[b]['web_url']
                if web_url not in already_done:
                    if 'abstract' in data[b].keys():
                        input_text = get_article_content(data[b], source='nyt')
                    else:
                        input_text = get_article_content(data[b], source='guardian')
                        
                    prompt = create_prompt_article_classification(input_text)
                  
                    prompts.append(prompt)
                    web_urls.append(web_url)
                    input_texts.append(input_text)
                    images.append(None)

            if len (input_texts)!=0:
                answers = generate_answer(images, prompts, tokenizer, model, template, args.max_tokens, args.use_vllm)[0]
                print(answers[0])
                for ans in range(len(answers)):
                    new_entry = {"web_url": web_urls[ans], "news_headline": input_texts[ans], "output": extract_first_list(answers[ans])}
                    results.append(new_entry)
            batch_count += args.batch_size
            # save results every 20 iterations
            if iter%20==0:
                output_file = os.path.join(root,f"{file}")
                with open(output_file, 'w') as  o_file:
                    json.dump(results, o_file, indent=4)

        #Save results
        output_file = os.path.join(root,f"{file}")
        with open(output_file, 'w') as  o_file:
            json.dump(results, o_file, indent=4)