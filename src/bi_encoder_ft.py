import os, math
import random
import argparse
from PIL import Image
from tqdm.auto import tqdm
import time 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import CLIPModel, CLIPProcessor
from transformers import get_cosine_schedule_with_warmup
from utils import *
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

def art_text(url):
    try:
        text = input_texts[article_caption_indices[url][0]]
    except:
        text = False
    return text


def _build_url_to_abstract_or_headline(articles):
    out = {}
    for a in articles:
        url = a.get("web_url")
        if not url: 
            continue
        txt = None
        if "abstract" in a:
            txt = a["abstract"]
        elif "fields" in a:
            txt = a['fields']["headline"]
        else:
            print(a)
        if txt:
            out[url] = txt
    return out


@torch.no_grad()
def encode_article_images(model, urls, resolver, processor, bs=128):
    feats, ids = [], []
    for s in tqdm(range(0, len(urls), bs)):
        batch_imgs, batch_ids = [], []
        for u in urls[s:s+bs]:
            img = resolver.open(u)
            if img is not None:
                batch_imgs.append(img); batch_ids.append(u)
        if not batch_imgs:
            continue
        batch = processor(images=batch_imgs, return_tensors="pt").to(device)
        emb   = model.get_image_features(**batch).float()
        feats.append(emb.cpu()); ids.extend(batch_ids)
    if not feats:
        return torch.empty(0, model.config.projection_dim), []
    feats = F.normalize(torch.cat(feats), dim=-1)
    return feats, ids


class PairCLIPDataset(Dataset):
    """
    Each item:
        img_path : str
        img_url  : str
        pos_urls : list[str]   (every caption URL that matches this image query based on the relevant articles set)
    """
    def __init__(self, instances, split,
                 time_url_map,            
                 url_to_caption,
                 seed=123):
        self.items = []
        self.rng = random.Random(seed)
        self.url2cap = url_to_caption 
        gt_urls = set()
        for inst in instances:
            img_url  = inst['web_url']
            pos_urls = [u for u in time_url_map[img_url]]
            if not pos_urls:
                continue
            self.items.append({
                "img_path": f"data/TARA_dataset/img/{split}/{inst['orig_idx']}.png",
                "img_url" : img_url,
                "pos_urls": set(pos_urls)
            })
            gt_urls.update(pos_urls)

        self.rand_neg_pool = sorted([u for u in url_to_caption.keys() if u not in gt_urls])
        self.url2owners = defaultdict(list)
        for it in self.items:
            for u in it["pos_urls"]:
                self.url2owners[u].append(it["img_url"])
        

    def __len__(self):       
        return len(self.items)
    def __getitem__(self, i): 
        return self.items[i]
    

def build_retrieval_eval_sets(
        dev,
        time_map,
        corpus_urls,                 
        article_caption_indices,  
        input_texts
):
    dev_img_infos = [
        (f"data/TARA_dataset/img/dev/{inst['orig_idx']}.png", inst['web_url'])
        for inst in dev
    ]

    url_to_caption = {}
    for url in corpus_urls:
        if url in article_caption_indices:                 # caption exists
            cap_idx = article_caption_indices[url][0]      
            url_to_caption[url] = input_texts[cap_idx]

    candidate_urls = sorted(url_to_caption.keys())    


    imgid_to_posurls = {
        img_url: {u for u in time_map[img_url]}
        for _, img_url in dev_img_infos
    }

    return dev_img_infos, candidate_urls, url_to_caption, imgid_to_posurls


def init_clip_partial_unfreeze(
        name = "openai/clip-vit-large-patch14",
        n_unfreeze= 4    
    ):
    model     = CLIPModel.from_pretrained(name).to(device)
    processor = CLIPProcessor.from_pretrained(name)


    for p in model.parameters():
        p.requires_grad = False

    #Unfreeze projection layers
    for p in model.visual_projection.parameters():
        p.requires_grad = True
    for p in model.text_projection.parameters():
        p.requires_grad = True
    # Unfreeze last n_unfreeze vision blocks
    n_layers_v = len(model.vision_model.encoder.layers)
    for idx, layer in enumerate(model.vision_model.encoder.layers):
        if idx >= n_layers_v - n_unfreeze:
            for p in layer.parameters():
                p.requires_grad = True
    # Unfreeze last n_unfreeze text blocks
    n_layers_t = len(model.text_model.encoder.layers)
    for idx, layer in enumerate(model.text_model.encoder.layers):
        if idx >= n_layers_t - n_unfreeze:
            for p in layer.parameters():
                p.requires_grad = True

    return model, processor


def clip_encode(model, pixel_values, input_ids, attention_mask):
    """
    Returns image and text embeddings with gradients enabled.
    """
    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    return out.image_embeds, out.text_embeds 


@torch.no_grad()
def encode_features(model, items, url_to_caption, processor, device, batch_size=64, as_text=False):
    model.eval()
    features = []
    ids = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start+batch_size]
        if not as_text:
            img_paths, img_urls = zip(*batch)
            img_list = [Image.open(p).convert("RGB") for p in img_paths]
            batch_out = processor(images=img_list, return_tensors="pt")
            pix = batch_out["pixel_values"].to(device)
            feats = model.get_image_features(pix).cpu()
            features.append(feats)
            ids.extend(img_urls)
        else:
            cap_urls = batch 
            cap_texts = [url_to_caption[u] for u in cap_urls]
            batch_out = processor(text=cap_texts, return_tensors="pt", padding=True, truncation=True)
            ids_tensor = batch_out["input_ids"].to(device)
            attn_tensor = batch_out["attention_mask"].to(device)
            feats = model.get_text_features(input_ids=ids_tensor, attention_mask=attn_tensor).cpu()
            features.append(feats)
            ids.extend(cap_urls)
    return torch.cat(features), ids


def evaluate_retrieval(model, dev_img_infos, candidate_urls, url_to_input, imgid_to_posurls, 
                       processor, device, k_vals=(1, 5, 10, 20, 50, 100)):
    img_feats, img_ids = encode_features(model, dev_img_infos, url_to_input, processor, device, as_text=False)
    img_feats = F.normalize(img_feats, dim=-1)


    txt_feats, cap_ids = encode_features(model, candidate_urls, url_to_input, processor, device, as_text=True)
    txt_feats = F.normalize(txt_feats, dim=-1)

    sims = img_feats @ txt_feats.T

    recalls = {}
    for k in k_vals:
        n_correct = 0
        for i, img_url in enumerate(img_ids):
            gt_caps = imgid_to_posurls[img_url]
            top_k = sims[i].topk(k).indices.tolist()
            top_k_cap_urls = [cap_ids[j] for j in top_k]
            if any(u in gt_caps for u in top_k_cap_urls):
                n_correct += 1
        recalls[f"R@{k}"] = n_correct / len(img_ids)
    return recalls


def build_url_to_article_relpath():
    records = []
    for file in os.listdir('data/processed_articles'):
        records += load_json(file)
    # Keep only entries that have both keys
    return {
        r['web_url']: r['image_path']
        for r in records
        if 'web_url' in r and 'image_path' in r
    }


class ArticleImageResolver:
    def __init__(self, url2rel, img_dirs):
        self.url2rel = url2rel             
        self.img_dirs = img_dirs            

    def open(self, url):
        rel = self.url2rel.get(url)
        if not rel:
            return None
        for d in self.img_dirs:
            #Search if image exists
            p = os.path.join(d, rel)
            if os.path.exists(p):
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    return None
        return None


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--base_ckpt", default="openai/clip-vit-large-patch14") 
    p.add_argument('--num_instances', type=int, default=12300)
    p.add_argument('--evidence_type', type=str, choices=['article','caption'], default='caption')
    p.add_argument('--num_cap', type=int, default=1, help="Number of captions to provide per article") 
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--recall_threshold',type=int,default=100)
    p.add_argument('--n_unfreeze',type=int,default=4)
    p.add_argument("--m_queries", type=int, default=64, help="number of query image per batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)


    args = p.parse_args()
    seed_everything(args.seed)

    train = load_jsonl('data/TARA_dataset/input/train.jsonl')[:args.num_instances] 
    dev = load_jsonl('data/TARA_dataset/input/gold_dev.jsonl')
    for i, inst in enumerate(train):
        inst['orig_idx'] = i
    for i, inst in enumerate(dev):
        inst['orig_idx'] = i
    train = [train[t] for t in range(len(train)) if t!=8569]
    gt_articles_train = load_json('data/relevant_articles_sets/relevant_articles_tara_train.json')
    gt_articles_dev = load_json('data/relevant_articles_sets/relevant_articles_tara_dev.json')
    train = [q for q in train if len(gt_articles_train[q['web_url']]['time']) > 0]
    dev   = [q for q in dev   if len(gt_articles_dev[q['web_url']]['time']) > 0]

    print('-------------------')
    print('Size of train and dev datasets')
    print(len(train))
    print(len(dev))
    print('-------------------')


    articles = []
    article_paths = 'data/processed_articles'
   
    for file in sorted(os.listdir(article_paths)):
        if ('2022' not in file) and ('2023' not in file) and ('guardian_part2' not in file):
            #do not include the data that is only meant for 5Pils
            articles += load_json(os.path.join(article_paths,file))
    qwen_class = []
    for file in sorted(os.listdir('data/qwen2.5_article_class')):
        if ('2022' not in file) and ('2023' not in file) and ('guardian_part2' not in file):
            qwen_class += load_json(os.path.join('data/qwen2.5_article_class', file))

    qwen_class = {q['web_url']:q['output'][0]  for q in qwen_class}
    # keep only category 1
    articles = [articles[a] for a in range(len(articles)) if qwen_class[articles[a]['web_url']].lower()=='category 1']
    #Remove all articles that do not have a keyword location
    articles = [a for a in articles if 'glocations' in [kw['name'].lower() for kw in a['keywords']]]
    #Add the tara train articles
    eval_articles = [a for a in articles]
    for file in ['data/tara_articles/train.json']:
        eval_articles += load_json(file)
    articles_urls_set = set([a['web_url'] for a in articles])
    eval_articles_urls_set = set([a['web_url'] for a in eval_articles])
    #Load news image captions generated by qwen2.5
    captions = []
    captions_paths = ["data/qwen2.5_caption_gt", "data/qwen2.5_caption"]
    for path in captions_paths:
        for file in os.listdir(f"{path}/news_image_caption/"):
            captions += load_json(os.path.join(f"{path}/news_image_caption/",file))
    captions = [c for c in captions if c['web_url'] in eval_articles_urls_set]
    web_urls= [c['web_url'] for c in captions for _ in range(min(len(c['output']), args.num_cap))]
    unique_web_urls = set(web_urls)

    article_caption_indices = {}
    for idx, url in enumerate(web_urls):
        article_caption_indices.setdefault(url, []).append(idx)
    articles = [a for a in articles if a['web_url'] in article_caption_indices.keys() and len(article_caption_indices[a['web_url']]) > 0]
    eval_articles = [a for a in eval_articles if a['web_url'] in article_caption_indices.keys() and len(article_caption_indices[a['web_url']]) > 0]
    if args.evidence_type=='caption':
        input_texts = [c['output'][idx] for c in captions for idx in range(min(len(c['output']), args.num_cap))]
    else:
        input_texts = []
        for a in articles:
            if 'abstract' in a.keys():
                input_texts.append(a['abstract'])
            else:
                #guardian
                input_texts.append(a['fields']['headline'])
    articles_urls_set = set([a['web_url'] for a in articles])
    eval_articles_urls_set = set([a['web_url'] for a in eval_articles])


    gt_articles_all = {**gt_articles_train, **gt_articles_dev}
    loc_url_map  = {u:set(v['location']) for u,v in gt_articles_all.items()}
    time_url_map = {u:set(v['time'])     for u,v in gt_articles_all.items()}


    # Build url to "input" (text OR image) for TRAIN and EVAL corpora
    url_to_input = {
        u: input_texts[idxs[0]]
        for u, idxs in article_caption_indices.items()
        if u in articles_urls_set and len(idxs) > 0
    }
    eval_url_to_input = {
        u: input_texts[idxs[0]]
        for u, idxs in article_caption_indices.items()
        if u in eval_articles_urls_set and len(idxs) > 0
    }

    candidate_urls         = sorted(url_to_input.keys())
    candidate_urls_dev_set = sorted(eval_url_to_input.keys())


    print('-----------------')
    print(f"Size of training candidates {len(candidate_urls)}")
    print(f"Size of evaluating candidates {len(candidate_urls_dev_set)}")
    print('-----------------')


    url_to_article_relpath = build_url_to_article_relpath()
    img_dirs_corpora = [
        "data/TARA_dataset/non_gt_img",
        "data/TARA_dataset/non_gt_img_guardian",
        "data/TARA_dataset/img/",
    ]
    article_resolver = ArticleImageResolver(url_to_article_relpath, img_dirs_corpora)


    model, processor = init_clip_partial_unfreeze(name=args.base_ckpt, n_unfreeze=args.n_unfreeze)

    # datasets and loaders
    train_ds = PairCLIPDataset(
    train, 'train',
    time_url_map=time_url_map,
    url_to_caption=url_to_input
)
    num_with_pos = sum(1 for it in train_ds.items if len(it["pos_urls"]) > 0)

    #Prepare negatives
    train_positive_urls = set()
    for img_url in time_url_map:
        train_positive_urls.update(time_url_map[img_url])
    train_positive_urls = {
        u for u in train_positive_urls if u in article_caption_indices
    }
    # corpus negatives
    pure_neg_pool = list(articles_urls_set - train_positive_urls)
    assert not any(u in train_positive_urls for u in pure_neg_pool), \
       "pure_neg_pool contains a training positive!"


    dev_img_infos, _, _, imgid_to_posurls = build_retrieval_eval_sets(dev,
                                                                        time_map=time_url_map,              
                                                                        corpus_urls=eval_articles_urls_set,
                                                                        article_caption_indices=article_caption_indices,
                                                                        input_texts=input_texts
                                                                        )



    def make_collate_square_hybrid(
        ds,
        m_queries,                 
        total_bs,                   
        resolver,                   
        rng=None
    ):
        """
        Builds a collate fn that:
        1) seeds M query images with non-conflicting positives
        2) tries to add hard negatives (owners first, else article image) if a fetcher is provided
        3) fills remaining slots with random negatives from ds.rand_neg_pool (article images)
        """
        if rng is None:
            rng = ds.rng

        def _collate(batch):
            imgs, caps, img_urls = [], [], []
            used_cap_urls = set()
            chosen_img_urls = set()
            chosen_pos = {}           # img_url -> chosen pos url
            forbidden = set()         # union of GT sets of chosen query images

            def _add_query_item(item, pos_url):
                img = Image.open(item["img_path"]).convert("RGB")
                imgs.append(img)
                caps.append(ds.url2cap[pos_url])
                img_urls.append(item["img_url"])
                chosen_img_urls.add(item["img_url"])
                chosen_pos[item["img_url"]] = pos_url
                used_cap_urls.add(pos_url)
                forbidden.update(item["pos_urls"])

            # 1) seed with M query images (non-conflicting positives)
            pool = batch[:]  # operate on a copy
            rng.shuffle(pool)
            for it in pool:
                if len(chosen_img_urls) >= m_queries:
                    break
                free = sorted(it["pos_urls"] - forbidden)
                if not free:
                    continue
                u = rng.choice(free)
                if u in used_cap_urls:
                    continue
                _add_query_item(it, u)

            # 1b) if not enough, pull more from whole ds
            if len(chosen_img_urls) < m_queries:
                pool2 = ds.items[:]
                rng.shuffle(pool2)
                for it in pool2:
                    if len(chosen_img_urls) >= m_queries:
                        break
                    if it["img_url"] in chosen_img_urls:
                        continue
                    free = sorted(it["pos_urls"] - forbidden)
                    if not free:
                        continue
                    u = rng.choice(free)
                    if u in used_cap_urls:
                        continue
                    _add_query_item(it, u)

            # 3) fill up with random negatives (article images only)
            if len(imgs) < total_bs:
                rand_pool = [u for u in ds.rand_neg_pool if (u not in used_cap_urls) and (u not in forbidden)]
                rng.shuffle(rand_pool)
                r = 0
                while len(imgs) < total_bs and r < len(rand_pool):
                    u = rand_pool[r]
                    r += 1
                    art_img = resolver.open(u)
                    if art_img is None:
                        #Do not use the article if it does not have an image
                        continue
                    imgs.append(art_img)
                    caps.append(ds.url2cap[u])
                    img_urls.append(f"[article]{u}")
                    used_cap_urls.add(u)

            if len(imgs) > total_bs:
                imgs, caps, img_urls = imgs[:total_bs], caps[:total_bs], img_urls[:total_bs]

            return imgs, caps, img_urls
    
        return _collate


    def build_square_loader(
        ds,
        batch_size,
        m_queries,
        resolver,
        torch_gen,
        seed
    ):
        def _seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        collate_fn = make_collate_square_hybrid(
            ds=ds,
            m_queries=m_queries,
            total_bs=batch_size,
            resolver=resolver,
            rng=ds.rng
        )

        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
            generator=torch_gen,
            worker_init_fn=_seed_worker,
            num_workers=0
        )

    

    torch_gen = torch.Generator().manual_seed(args.seed)

    train_loader = build_square_loader(
        train_ds,
        batch_size=args.bs,         
        m_queries=args.m_queries,
        resolver=article_resolver,
        torch_gen=torch_gen,
        seed=args.seed,
        hard_neg_fetcher=None
    )
    img2pos = {it["img_url"]: set(it["pos_urls"]) for it in train_ds.items}


    url2owners = defaultdict(list)
    for iu, pos in img2pos.items():
        for u in pos:
            url2owners[u].append(iu)
    train_ds.url2owners = url2owners

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.02)
    steps_per_epoch = len(train_loader)     
    opt_steps_per_epoch = math.ceil(steps_per_epoch / max(1, args.grad_accum))
    num_train_steps = opt_steps_per_epoch * args.epochs
    num_epochs      = args.epochs
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=500, num_training_steps=num_train_steps)
    #Provide initial results
    start_time = time.time()
    recalls = evaluate_retrieval(model, dev_img_infos, candidate_urls_dev_set, eval_url_to_input,
                                imgid_to_posurls, processor, device,
                                 resolver=article_resolver)
    elapsed = time.time() -start_time
    print(f"Full dev evaluation took {elapsed:.1f} seconds")
    print("Full corpus recall")
    print(recalls)
    best_r = 0.0
    if recalls[f'R@{args.recall_threshold}'] > best_r:
        best_r= recalls[f'R@{args.recall_threshold}']

    for epoch in range(args.epochs):
        train_loader = build_square_loader(
            ds=train_ds,
            batch_size=args.bs,
            m_queries=args.m_queries,
            resolver=article_resolver,
            torch_gen=torch_gen,
            seed=args.seed,
            hard_neg_fetcher=None
        )


        model.train()
        print_c = 0
        opt.zero_grad(set_to_none=True)
        for step, (imgs, caps, _) in enumerate(tqdm(train_loader, desc=f"E{epoch}"), start=1):
            if print_c == 0:
                print(f"Number of images {len(imgs)}")
                print(f"Number of captions {len(caps)}")
                print_c +=1
            batch = processor(images=imgs, text=caps, return_tensors="pt",
                            padding=True, truncation=True)

            pix = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            img_emb = F.normalize(model.get_image_features(pix), dim=-1)    
            txt_emb = F.normalize(model.get_text_features(ids, att), dim=-1) 

            # similarity matrix and InfoNCE loss
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * (img_emb @ txt_emb.T)                   

            #two-way NCE
            targets = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, targets) +
                    F.cross_entropy(logits.T, targets)) / 2

            loss = loss / max(1, args.grad_accum)
            loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.logit_scale.clamp_(math.log(1.0), math.log(50.0))

        last_incomplete = (step % args.grad_accum) != 0
        if last_incomplete:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            opt.zero_grad(set_to_none=True)


        # evaluate
        start_time = time.time()
        recalls = evaluate_retrieval(model, dev_img_infos, candidate_urls_dev_set, eval_url_to_input, imgid_to_posurls, processor, device, resolver=article_resolver)
        elapsed = time.time() -start_time
        print(f"Full dev evaluation took {elapsed:.1f} seconds")
        print("Full corpus recall")
        print(recalls)
        os.makedirs('clip_model',exist_ok=True)
        if recalls[f'R@{args.recall_threshold}'] > best_r:
            #A better model has been obtained this epoch
            best_r = recalls[f'R@{args.recall_threshold}']
            output_path = f"clip_model/bi_encoder.pt"
            torch.save(model.state_dict(), output_path)
            print(f"  → Saved {output_path}")

        print(f"Training complete. Best R@{args.recall_threshold}:", best_r)
    