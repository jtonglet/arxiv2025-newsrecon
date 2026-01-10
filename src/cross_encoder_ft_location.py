import os, argparse, random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections import defaultdict, Counter
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


# Pair dataset 
class PairDS(Dataset):
    def __init__(self, img_infos, sims, url_ids, gt_map, url2loc, url2text, top_k=100, max_negs=5):
        """
        Stores (img_path, text_for_ce, label).
        Negatives have no overlap in location atoms with the chosen positive.
        Also enforces unique location-text among negatives for diversity
        """
        self.items = []
        assert sims.shape[1] == len(url_ids), "sims and url_ids mismatch"

        atoms_cache = {}
        def atoms_for_loc_text(txt: str) -> set:
            if txt not in atoms_cache:
                atoms_cache[txt] = extract_loc_atoms(txt)
            return atoms_cache[txt]

        for i, (img_path, img_url) in enumerate(img_infos):
            pos_urls = set(gt_map.get(img_url, set()))
            top_idx = sims[i].topk(top_k).indices.tolist()
            cand = [url_ids[j] for j in top_idx if url_ids[j] in url2loc and url_ids[j] in url2text]

            pos = [u for u in cand if u in pos_urls]
            neg = [u for u in cand if u not in pos_urls]
            if not pos or not neg:
                continue

            pos_u = random.choice(pos) #take one positive
            pos_atoms = atoms_for_loc_text(url2loc[pos_u])

            neg_far = [u for u in neg if atoms_for_loc_text(url2loc[u]).isdisjoint(pos_atoms)]
            if not neg_far:
                continue

            groups = defaultdict(list)
            for u in neg_far:
                groups[url2loc[u]].append(u)

            chosen_urls = [pos_u]
            used_loc_texts = set()

            most_common = [lt for lt, _ in Counter({lt: len(vs) for lt, vs in groups.items()}).most_common(2)]
            for lt in most_common:
                if lt in used_loc_texts:
                    continue
                u = random.choice(groups[lt])
                chosen_urls.append(u)
                used_loc_texts.add(lt)

            remaining_lts = [lt for lt in groups.keys() if lt not in used_loc_texts]
            random.shuffle(remaining_lts)
            for lt in remaining_lts:
                if len(chosen_urls) >= 1 + max_negs:
                    break
                u = random.choice(groups[lt])
                chosen_urls.append(u)
                used_loc_texts.add(lt)

            for u in chosen_urls:
                y = float(u in pos_urls)
                self.items.append((img_path, url2text[u], y))

    def __len__(self):  return len(self.items)
    def __getitem__(self, idx):
        p, txt, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return img, txt, torch.tensor(y, dtype=torch.float32)


# Cross-encoder MLP head
class CrossEncMLP(nn.Module):

    def __init__(self, clip_base, hidden=[256], operations="concatenation"):
        super().__init__()
        self.clip = clip_base
        self.operations = operations

        # freeze all
        for p in self.clip.parameters():
            p.requires_grad = False

        D = self.clip.config.projection_dim
        # input dim: 2D for "concatenation" (img || txt), +D for each of "multiplication" and "difference"
        in_dim = 0
        if "concatenation" in self.operations:
            in_dim += 2 * D
        if "multiplication" in self.operations:
            in_dim += D
        if "difference" in self.operations:
            in_dim += D

        layers = [nn.Linear(in_dim, hidden[0]), nn.GELU()]
        layers.append(nn.Linear(hidden[0], 1))
        self.head = nn.Sequential(*layers)

    def forward(self, pixel_values, input_ids, attention_mask):
        img_emb = F.normalize(self.clip.get_image_features(pixel_values=pixel_values), dim=-1)
        txt_emb = F.normalize(self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask), dim=-1)

        pieces = []
        if "concatenation" in self.operations:
            pieces += [img_emb, txt_emb]
        if "multiplication" in self.operations:
            pieces.append(img_emb * txt_emb)
        if "difference" in self.operations:
            pieces.append((img_emb - txt_emb).abs())

        feat = torch.cat(pieces, dim=-1)
        return self.head(feat).squeeze(1)

def build_crossenc(clip_base, **kwargs):
    return CrossEncMLP(
        clip_base,
        hidden=[256],
        operations=kwargs.get("operations", ["concatenation"])
    )

# Eval helpers 
@torch.no_grad()
def eval_rerank(
    model, processor,
    img_infos, cands, gt_map,
    url2text,                
    K_list=(1, 5, 10),
    bs=128
):
    """
    Re-rank using the SAME text used in training (url2text).
    """
    model.eval()
    Kmax = max(K_list)

    total = len(img_infos)
    base_hits_total   = {K: 0 for K in K_list}
    rerank_hits_total = {K: 0 for K in K_list}
    covered_count     = {K: 0 for K in K_list}

    for (img_path, img_url) in tqdm(img_infos, desc="rerank dev"):
        full_list = cands.get(img_url, [])
        if not full_list:
            continue

        cand_urls = full_list[:Kmax]
        gt_urls = gt_map.get(img_url, set())

        # Baseline BI coverage/success at each K
        for K in K_list:
            topK_bi = cand_urls[:K]
            if any(u in gt_urls for u in topK_bi):
                covered_count[K] += 1
                base_hits_total[K] += 1

        if not cand_urls:
            continue

        img = Image.open(img_path).convert("RGB")
        texts = [url2text[u] for u in cand_urls] 

        scores = []
        for s in range(0, len(cand_urls), bs):
            batch = processor(
                images=[img] * min(bs, len(cand_urls) - s),
                text=texts[s:s+bs],
                return_tensors="pt", padding=True, truncation=True
            ).to(device)
            logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            scores.append(torch.sigmoid(logits).float().cpu())
        scores = torch.cat(scores, dim=0).numpy().tolist()

        order = sorted(range(len(cand_urls)), key=lambda i: scores[i], reverse=True)
        reranked = [cand_urls[i] for i in order]

        for K in K_list:
            topK_ce = reranked[:K]
            if any(u in gt_urls for u in topK_ce):
                rerank_hits_total[K] += 1

    def div(a, b): return (a / b) if b else 0.0
    return {
        "total": total,
        "coverage": {f"R@{K}": div(covered_count[K], total) for K in K_list},
        "baseline_global": {f"R@{K}": div(base_hits_total[K], total) for K in K_list},
        "rerank_global": {f"R@{K}": div(rerank_hits_total[K], total) for K in K_list},
    }


def eval_classifier(model, loader, processor):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, caps, y in tqdm(loader):
            batch = processor(images=imgs, text=caps, return_tensors="pt",
                              padding=True, truncation=True).to(device)
            logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            pred = (torch.sigmoid(logits) > 0.5).long().cpu()
            all_preds.append(pred)
            all_true.append(y.long().cpu())
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_true).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    prec= precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return {"acc": acc, "prec": prec, "rec": recall, "f1_macro": f1}

def subset_random(ds, size_or_frac):
    if not size_or_frac or size_or_frac <= 0:
        return ds
    n = len(ds)
    m = int(size_or_frac * n) if size_or_frac < 1 else min(int(size_or_frac), n)
    idx = random.sample(range(n), m)
    return torch.utils.data.Subset(ds, idx)

def sample_train_balanced(ds, size_or_frac):
    if not size_or_frac or size_or_frac <= 0 or size_or_frac==1:
        return ds
    base_items = ds.items if isinstance(ds, PairDS) else ds.dataset.items
    if isinstance(ds, PairDS):
        indices = list(range(len(base_items)))
    else:
        indices = list(ds.indices)
    pos_idx = [i for i in indices if int(base_items[i][2]) == 1]
    neg_idx = [i for i in indices if int(base_items[i][2]) == 0]
    total = len(indices)
    target_total = max(2, int(size_or_frac * total)) if size_or_frac < 1 else min(int(size_or_frac), total)
    if target_total % 2 == 1:
        target_total -= 1
    target_total = min(target_total, 2 * min(len(pos_idx), len(neg_idx)))
    half = max(1, target_total // 2)
    pos_sel = random.sample(pos_idx, half)
    neg_sel = random.sample(neg_idx, half)
    chosen = pos_sel + neg_sel
    random.shuffle(chosen)
    if isinstance(ds, PairDS):
        return torch.utils.data.Subset(ds, chosen)
    else:
        return torch.utils.data.Subset(ds.dataset, chosen)

#Main 
def main(args):
    random.seed(42)

    gt_train = load_json("data/relevant_articles_sets/relevant_articles_tara_train.json")
    gt_dev   = load_json("data/relevant_articles_sets/relevant_articles_tara_dev.json")
    gt_map  = {u:set(v["location"]) for u,v in {**gt_train, **gt_dev}.items()}

    # Image splits
    train_q = load_jsonl("data/TARA_dataset/input/train.jsonl")
    dev_q   = load_jsonl("data/TARA_dataset/input/gold_dev.jsonl")
    for s in (train_q, dev_q):
        for i,x in enumerate(s): x["orig_idx"]=i
    train_q = [train_q[t] for t in range(len(train_q)) if t!=8569]

    # Articles
    articles = []
    article_paths = 'data/processed_articles'
    for file in sorted(os.listdir(article_paths)):
        if ('2022' not in file) and ('2023' not in file) and ('guardian_part2' not in file):
            articles += load_json(os.path.join(article_paths,file))
    qwen_class = []
    for file in sorted(os.listdir('data/qwen2.5_article_class')):
        if ('2022' not in file) and ('2023' not in file) and ('guardian_part2' not in file):
            qwen_class += load_json(os.path.join('data/qwen2.5_article_class', file))
    qwen_class = {q['web_url']:q['output'][0]  for q in qwen_class}
    # keep only category 1
    articles = [articles[a] for a in range(len(articles)) if qwen_class[articles[a]['web_url']].lower()=='category 1']
    # require at least one glocations keyword
    articles = [a for a in articles if 'glocations' in [kw['name'].lower() for kw in a['keywords']]]
    # Add tara train articles to eval pool (then filter)
    eval_articles = [a for a in articles]
    for file in ['data/tara_articles/train.json']:
        eval_articles += load_json(file)
    eval_articles_urls_set = set([a['web_url'] for a in eval_articles])

    # Captions
    captions = []
    captions_paths = ["data/qwen2.5_caption_gt", "data/qwen2.5_caption"]
    for path in captions_paths:
        for file in os.listdir(f"{path}/news_image_caption/"):
            captions += load_json(os.path.join(f"{path}/news_image_caption/",file))
    captions = [c for c in captions if c['web_url'] in eval_articles_urls_set]

    web_urls= [c['web_url'] for c in captions for _ in range(min(len(c['output']), 1))]
    article_caption_indices = {}
    for idx, url in enumerate(web_urls):
        article_caption_indices.setdefault(url, []).append(idx)

    # keep only articles that have at least one caption
    articles = [a for a in articles if a['web_url'] in article_caption_indices and len(article_caption_indices[a['web_url']]) > 0]
    eval_articles = [a for a in eval_articles if a['web_url'] in article_caption_indices and len(article_caption_indices[a['web_url']]) > 0]
    articles_urls_set = set([a['web_url'] for a in articles])
    eval_articles_urls_set = set([a['web_url'] for a in eval_articles])

    # Build maps
    url2cap = build_url2cap_first(captions)
    url2loc_text = build_url2loc_text_from_articles(articles, allowed_urls=articles_urls_set)
    cand_urls = sorted(set(url2cap.keys()) & set(url2loc_text.keys()))


    url2text = {u: f"An image from {url2loc_text[u]}" for u in cand_urls}

    # Load CLIP retriever (bi-encoder)
    base_ckpt = args.base_ckpt
    ckpt_name = args.biencoder_ckpt
    cache_dir = args.cache_dir
    clip_retriever = CLIPModel.from_pretrained(base_ckpt).to(device)
    if args.biencoder_ckpt != 'base_model':
        state = torch.load(f"clip_model/{args.biencoder_ckpt}", map_location=device)
        clip_retriever.load_state_dict(state, strict=False)
        clip_retriever.eval()
    processor = CLIPProcessor.from_pretrained(base_ckpt)

    # Encode CAPTIONS for candidate selection
    cap_txt_feats, cap_url_ids = cached_encode_caps_for_urls(
        clip_retriever, cand_urls, url2cap, processor,
    bs=256, cache_dir=cache_dir, ckpt_name=ckpt_name, model_id=base_ckpt
    )

    def build_pairs_and_cands(split_name, top_k, recompute_cache):
        img_infos = [
            (f"data/TARA_dataset/img/{split_name}/{q['orig_idx']}.png", q['web_url'])
            for q in (train_q if split_name == "train" else dev_q)
        ]
        img_feats, _ = cached_encode_imgs(
            clip_retriever, img_infos, processor,
            bs=64, cache_dir=cache_dir, ckpt_name=ckpt_name, model_id=base_ckpt,
            tag=f"{split_name}_imgs", recompute_cache=recompute_cache
        )
        sims = img_feats @ cap_txt_feats.T
        cands = {}
        for i, (_, img_url) in enumerate(img_infos):
            top_idx = sims[i].topk(top_k).indices.tolist()
            kept = []
            for j in top_idx:
                u = cap_url_ids[j]
                if u in url2loc_text and u in url2text:
                    kept.append(u)
            cands[img_url] = kept
        ds = PairDS(img_infos, sims, cap_url_ids, gt_map, url2loc_text, url2text, top_k=top_k)
        return ds, img_infos, sims, cands


    train_ds, _, _, _ = build_pairs_and_cands("train", args.top_k, args.recompute_cache)
    dev_ds,   dev_img_infos, _ , dev_cands = build_pairs_and_cands("dev", args.top_k, args.recompute_cache)
    train_ds = sample_train_balanced(train_ds, 1)
    dev_ds   = subset_random(dev_ds, 1)

    def collate_pairs(batch):
        imgs, caps, ys = zip(*batch)
        return list(imgs), list(caps), torch.stack(ys, dim=0)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate_pairs)
    dev_loader   = DataLoader(dev_ds  , batch_size=args.bs*2, shuffle=False, num_workers=4, collate_fn=collate_pairs)

    # Free the bi-encoder model from memory
    del clip_retriever
    torch.cuda.empty_cache()

    # Fresh CLIP for CE
    clip_ce = CLIPModel.from_pretrained(base_ckpt).to(device)
    clip_ce.eval()

    # Build CE model
    model = build_crossenc(
        clip_ce,
        operations=args.operations.split("-")
    ).to(device)

    head_params = list(model.head.parameters())
    trainable_params= [{"params": head_params, "lr": args.lr}]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    pos_w = get_pos_weight(train_ds).to(device)
    bce   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    #training loop
    best_r1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"epoch {epoch}")
        for imgs, caps, y in loop:
            batch = processor(images=imgs, text=caps, return_tensors="pt",
                              padding=True, truncation=True).to(device)
            y = y.to(device)
            logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loop.set_postfix(loss=float(loss))

        #  dev evaluation
        metrics = eval_classifier(model, dev_loader, processor)
        print(f"dev — acc: {metrics['acc']:.3f}  F1(macro): {metrics['f1_macro']:.3f}  "
              f"prec: {metrics['prec']:.3f}  rec: {metrics['rec']:.3f}")

        ret = eval_rerank(
            model, processor, dev_img_infos, dev_cands, gt_map,
            url2text=url2text,    
            K_list=(1,5,10)
        )

        r1 = ret['rerank_global']['R@1']
        if r1 > best_r1 and r1 >= ret['baseline_global']['R@1']:
            best_r1 = r1
            model_output_path = f"clip_model/cross_encoder_location.pt"

            torch.save(model.head.state_dict(), model_output_path)
            print(f"saved {model_output_path}")

    print("Best dev R@1:", best_r1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", default="openai/clip-vit-large-patch14") 
    ap.add_argument("--biencoder_ckpt", default="bi_encoder.pt")
    ap.add_argument("--top_k",  type=int, default=20)
    ap.add_argument("--bs",     type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--operations", type=str, default="concatenation", help="Use '-' to combine: concatenation-multiplication-difference")
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--cache_dir", type=str, default="cache_clip_embeds",
                    help="Directory to store and load cached bi encoder CLIP embeddings.")
    ap.add_argument("--recompute_cache", default=0, type=int,
                help="Force recomputation and overwrite cache files.")
    ap.add_argument("--seed", choices=[123,456,789], type=int)
    args = ap.parse_args()
    seed_everything(args.seed)
    main(args)