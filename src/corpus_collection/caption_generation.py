import argparse
import os
from tqdm import tqdm
from utils import *
from llm_inference import *
from loaders import *
from prompts import create_prompt_captioning
import random


SEED = 42
random.seed(SEED)

# python caption_generation.py  --input_folder data/processed_articles
# python caption_generation.py  --input_folder data/tara_articles

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='data/processed_articles', choices=['data/processed_articles', 'data/tara_articles'],
                         help="Folder containing the article files")
    parser.add_argument('--model', type=str, default='qwen2.5/7B/',  help="Name of the model to run")
    parser.add_argument('--use_vllm', type=int, default = 1, help="set to 1 to use vllm for inference")
    parser.add_argument('--batch_size', type=int, default = 64, help="Number of prompts to instances to provide to the llm in one batch")
    parser.add_argument('--max_tokens', type=int, default=256,  help="Max number of generated tokens")
    parser.add_argument('--num_output', type=int, default=5, help="Number of output captions to generate")
    
    args = parser.parse_args()

    if args.input_folder=='data/processed_articles':
        output_folder = "data/qwen2.5_caption"
    else:
        output_folder = "data/qwen2.5_caption_gt"
    root = os.path.join(output_folder, "news_image_caption")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(root, exist_ok=True)

    #Load the model
    template = args.model
    model, tokenizer = load_model(template, use_vllm=args.use_vllm) 

    #Load the articles
    for file in os.listdir(args.input_folder):
        data = load_json(os.path.join(args.input_folder, file))

        if os.path.exists(os.path.join(root,f"{file}")):
            # Verify whether captions have already been generated for these articles
            results = load_json(os.path.join(root,f"{file}"))
            already_done = set([r['web_url'] for r in results])
            print(f"Loading {len(already_done)} already captioned instances")
        else:
            results = []
            already_done = set()
        print('--------------------------------')
        print(f'Before removing already done {len(data)}')
        qwen_class = load_json(os.path.join('data/qwen2.5_article_class', file))
        #Remove articles from  catagory 2
        data = [data[d] for d in range(len(data)) if data[d]['web_url'] not in already_done and qwen_class[d]['output'][0].lower()=='category 1']
        print(f'After removing already done and category 2 articles {len(data)}')
        print('--------------------------------')
        batch_count = 0
        num_batches = (len(data) + args.batch_size - 1) // args.batch_size
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
                    prompt = create_prompt_captioning(input_text, args.num_output, args.task)
                    prompts.append(prompt)
                    web_urls.append(web_url)
                    input_texts.append(input_text)
                    images.append(None)

            if len (input_texts)!=0:
                answers = generate_answer(images, prompts, tokenizer, model, template, args.max_tokens, args.use_vllm)
                if len(answers)==2 and answers[1]==0:
                    answers = answers[0]
                for ans in range(len(answers)):
                    new_entry = {"web_url": web_urls[ans], "output": extract_first_list(str(answers[ans]))}
                    results.append(new_entry)
            batch_count += args.batch_size

        #Save results
        output_file = os.path.join(root,f"{file}")
        with open(output_file, 'w') as  o_file:
            json.dump(results, o_file, indent=4)