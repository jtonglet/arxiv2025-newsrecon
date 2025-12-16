import os, argparse, random, re, datetime
from datetime import datetime, timedelta
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_url2date(articles_list, allowed_urls):
    """
    Map web_url -> YYYY-MM-DD (parsed from article['pub_date'] if present).
    Accepts full ISO timestamps and trims to date.
    """
    url2date = {}
    for a in articles_list:
        url = a.get("web_url")
        if not url or url not in allowed_urls:
            continue
        d = a.get("pub_date") or a.get("pubdate") or a.get("date")
        if not d:
            continue
        # keep just the date part if timestamp
        # examples: "2021-06-02T13:45:00+0000" -> "2021-06-02"
        m = re.match(r"(\d{4}-\d{2}-\d{2})", str(d))
        if m:
            url2date[url] = m.group(1)
    return url2date


def build_time_clusters_for_image(
    sims_row, cap_url_ids, top_k, url2loc_text, url2date, gt_urls_set, Nmin, Nwindow
):
    """
    Build time clusters for one query image from its bi-encoder similarity row.

    A valid cluster must:
      - have >= Nmin articles
      - share at least ONE identical location keyword across ALL its articles
      - satisfy max(date) - min(date) <= 15 days  (=> all pairwise <= 15)
    The cluster's CE text uses that shared keyword and the window [min_date, max_date].
    """
    # Top-K candidate URLs that have both location keywords and a date
    top_idx = sims_row.topk(top_k).indices.tolist()
    cand = [cap_url_ids[j] for j in top_idx if cap_url_ids[j] in url2loc_text and cap_url_ids[j] in url2date]
    if not cand:
        return []

    # Precompute per-URL info: date + normalized location-token set
    def _parse_date(s):
        return datetime.strptime(s, "%Y-%m-%d").date()
    
    info = {
        u: {
            "date": _parse_date(url2date[u]),
            "tokens": extract_loc_atoms(url2loc_text[u]) 
        } for u in cand
    }

    # Index candidates by token (only tokens that actually appear)
    token_to_urls = {}
    for u in cand:
        for t in info[u]["tokens"]:
            token_to_urls.setdefault(t, []).append(u)

    # For each token, create sliding windows in date order with span <= 15 days
    clusters = []
    seen_keys = set()  # deduplicate across tokens
    for token, urls in token_to_urls.items():
        if len(urls) < Nmin:
            continue
        urls_sorted = sorted(urls, key=lambda x: info[x]["date"])

        left = 0
        for right in range(len(urls_sorted)):
            # shrink left until window span <= 15 days
            window = 2 * Nwindow +1
            while info[urls_sorted[right]]["date"] - info[urls_sorted[left]]["date"] > timedelta(days=window):
                left += 1
            if right - left + 1 >= Nmin:
                group = urls_sorted[left:right+1]

                key = tuple(sorted(group))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                dmin, dmax = info[group[0]]["date"], info[group[-1]]["date"]
                dmin_txt = dmin.isoformat()
                dmax_txt = dmax.isoformat()
                text = f"An image taken between {dmin_txt} and {dmax_txt} in {token}"
                label = any(u in gt_urls_set for u in group)

                clusters.append({
                                    "urls": group,
                                    "text": text,
                                    "label": float(label),
                                    "token": token,              # shared location keyword
                                    "dmin": dmin.isoformat(),    # earliest date in cluster
                                    "dmax": dmax.isoformat(),    # latest date in cluster
                                })
    return clusters


#  Pair dataset 
class PairDSClusters(Dataset):
    def __init__(self, img_infos, sims, url_ids, gt_map, url2loc, url2date,
                 top_k=100, Nmin=2, Nwindow=7, max_negs=5):
        self.items = []
        for i, (img_path, img_url) in enumerate(img_infos):
            gt_urls = set(gt_map.get(img_url, set()))
            clusters = build_time_clusters_for_image(
                sims[i], url_ids, top_k, url2loc, url2date, gt_urls, Nmin, Nwindow
            )
            if not clusters: 
                continue
            pos = [cl for cl in clusters if cl["label"] == 1.0]
            neg = [cl for cl in clusters if cl["label"] == 0.0]
            if not pos or not neg:
                continue  # require ≥1 pos and ≥1 neg
            from datetime import datetime
            def _parse(d):  # 'YYYY-MM-DD' -> date
                return datetime.strptime(d, "%Y-%m-%d").date()

            def _overlap(c1, c2):
                a1, a2 = _parse(c1["dmin"]), _parse(c1["dmax"])
                b1, b2 = _parse(c2["dmin"]), _parse(c2["dmax"])
                return not (a2 < b1 or b2 < a1)

            # pick 1 positive
            pos_c = random.choice(pos)

            # select non-redundant, article-disjoint negatives
            chosen = [pos_c]
            chosen_urls = set(pos_c["urls"])
            # Use a local seen-key set to avoid duplicates within this image
            seen_keys = set()
            # Seed with positive cluster key
            pkey = (pos_c.get("token", ""), pos_c.get("dmin", ""), pos_c.get("dmax", ""), pos_c["text"])
            seen_keys.add(pkey)

            random.shuffle(neg)
            for neg_c in neg:
                # basic article disjointness
                if set(neg_c["urls"]) & chosen_urls:
                    continue
                redundant = False
                for already in chosen[1:]:  # compare only with negatives already kept
                    if already.get("token") == neg_c.get("token") and _overlap(already, neg_c):
                        redundant = True
                        break
                if redundant:
                    continue
                nkey = (neg_c.get("token", ""), neg_c.get("dmin", ""), neg_c.get("dmax", ""), neg_c["text"])
                if nkey in seen_keys:
                    continue
                chosen.append(neg_c)
                chosen_urls.update(neg_c["urls"])
                seen_keys.add(nkey)
                if len(chosen) >= 1 + max_negs:
                    break
            for cl in chosen:
                self.items.append((img_path, cl["text"], torch.tensor(cl["label"], dtype=torch.float32)))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p, txt, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        return img, txt, y


# Cross-encoder MLP hea
class CrossEncMLP(nn.Module):
    """
    Frozen (optionally partially unfrozen) CLIP towers.
    Head sees pooled CLIP embeddings (concatenated features per chosen operations).
    """
    def __init__(self, clip_base, hidden=[256], dropout=0.1, operations="concatenation"):
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
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
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
        dropout=kwargs.get("dropout", 0.1),
        operations=kwargs.get("operations", ["concatenation"])
    )


# Eval helpers
@torch.no_grad()
def eval_time_clusters_recall_at_1(
    model, processor,
    img_infos, sims, cap_url_ids, cands, gt_map,
    top_k=100, bs=128, min_clusters=2
):
    model.eval()
    total = len(img_infos)

    # Covered (>= min_clusters)
    covered = 0
    covered_pos = 0
    base_hits_cov = 0
    ce_hits_cov = 0

    # Uncovered (< min_clusters)
    notcov = 0
    bi_hits_notcov = 0

    for i, (img_path, img_url) in enumerate(tqdm(img_infos, desc="eval R@1 (time clusters)")):
        clusters = cands.get(img_url, [])
        gt_urls = set(gt_map.get(img_url, set()))

        # BI top1
        top_idx = sims[i].topk(top_k).indices.tolist()
        top1_url = cap_url_ids[top_idx[0]] if top_idx else None

        if len(clusters) < min_clusters:
            # Uncovered
            notcov += 1
            if top1_url in gt_urls:
                bi_hits_notcov += 1
            continue
        # Covered branch
        covered += 1

        # Identify positive clusters (any URL in GT)
        pos_cluster_idx = [k for k, c in enumerate(clusters) if any(u in gt_urls for u in c["urls"])]
        if pos_cluster_idx:
            covered_pos += 1
        pos_urls_union = set()
        for k in pos_cluster_idx:
            pos_urls_union.update(clusters[k]["urls"])

        # Baseline R@1 on covered
        if top1_url in pos_urls_union:
            base_hits_cov += 1

        # CE R@1 on covered: pick highest-CE cluster, check if it's positive
        img = Image.open(img_path).convert("RGB")
        texts = [c["text"] for c in clusters]
        scores = []
        for s in range(0, len(texts), bs):
            batch = processor(
                images=[img] * min(bs, len(texts) - s),
                text=texts[s:s+bs],
                return_tensors="pt", padding=True, truncation=True
            ).to(device)
            logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            scores.append(torch.sigmoid(logits).float().cpu())
        scores = torch.cat(scores, dim=0).numpy().tolist()
        best_i = max(range(len(scores)), key=lambda j: scores[j])
        if best_i in pos_cluster_idx:
            ce_hits_cov += 1

    def div(a, b): return (a / b) if b else 0.0

    # Covered-only metrics
    oracle_cov = div(covered_pos, covered)
    baseline_cov = div(base_hits_cov, covered)
    ce_cov = div(ce_hits_cov, covered)
    ce_over_oracle = div(ce_cov, oracle_cov) if oracle_cov > 0 else 0.0

    # Uncovered BI recall
    bi_notcov = div(bi_hits_notcov, notcov)

    # Overall metrics
    overall_baseline = div(bi_hits_notcov + base_hits_cov, total)
    overall_ce       = div(bi_hits_notcov + ce_hits_cov,   total)
    overall_oracle   = div(bi_hits_notcov + covered_pos,   total)

    return {
        # coverage breakdown
        "total": total,
        "covered": covered,
        "not_covered": notcov,
        "covered_pos": covered_pos,
        # covered-only recalls
        "oracle_R1": oracle_cov,
        "baseline_R1": baseline_cov,
        "ce_R1": ce_cov,
        "ce_over_oracle": ce_over_oracle,
        # uncovered BI
        "bi_R1_uncovered": bi_notcov,
        # overall recalls starting from the BI base on uncovered split
        "overall_baseline_R1": overall_baseline,
        "overall_ce_R1": overall_ce,
        "overall_oracle_R1": overall_oracle,
    }


def subset_dev(img_infos, sims, cands, frac):
    if not frac or frac <= 0 or frac >= 1:
        return img_infos, sims, cands
    m = int(len(img_infos) * frac)
    idxs = random.sample(range(len(img_infos)), m)
    img_infos_sub = [img_infos[i] for i in idxs]
    sims_sub = sims[idxs]
    cands_sub = {img_infos[i][1]: cands[img_infos[i][1]] for i in idxs}
    return img_infos_sub, sims_sub, cands_sub

def sample_train_balanced(ds, size_or_frac):
    if not size_or_frac or size_or_frac <= 0 or size_or_frac>=1:
        return ds
    base_items = ds.items if isinstance(ds, PairDSClusters) else ds.dataset.items
    if isinstance(ds, PairDSClusters):
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
    if isinstance(ds, PairDSClusters):
        return torch.utils.data.Subset(ds, chosen)
    else:
        return torch.utils.data.Subset(ds.dataset, chosen)


# Main function
def main(args):
    random.seed(42)

    # Relevant articles set
    gt_train = load_json("data/gt_articles_sets/gt_articles_tara_train.json")
    gt_dev   = load_json("data/gt_articles_sets/gt_articles_tara_dev.json")
    gt_map  = {u:set(v["time"]) for u,v in {**gt_train, **gt_dev}.items()}

    # Image splits
    train_q = load_jsonl("data/TARA_dataset/input/train.jsonl")
    dev_q   = load_jsonl("data/TARA_dataset/input/gold_dev.jsonl")
    for s in (train_q, dev_q):
        for i,x in enumerate(s): x["orig_idx"]=i
    #Instance 8569 of the train set is removed because the image could not be scraped
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
    qwen_class = {q['web_url']:q['output'][0] for q in qwen_class}
    # keep only category 1
    articles = [articles[a] for a in range(len(articles)) if qwen_class[articles[a]['web_url']].lower()=='category 1']
    # require at least one glocations keyword
    articles = [a for a in articles if 'glocations' in [kw['name'].lower() for kw in a['keywords']]]

    # Add tara articles train set
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
    url2date = build_url2date(articles, allowed_urls=articles_urls_set)

    # Candidate pool
    cand_urls = sorted(set(url2cap.keys()) & set(url2loc_text.keys()) & set(url2date.keys()))

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


        Nmin = args.N_min
        Nwindow = args.N_window
        ds = PairDSClusters(
            img_infos, sims, cap_url_ids, gt_map, url2loc_text, url2date,
            top_k=top_k, Nmin=Nmin, Nwindow= Nwindow, max_negs=5
        )

        # For eval/dev candidates
        cands = {}
        for i, (_, img_url) in enumerate(img_infos):
            gt_urls = set(gt_map.get(img_url, set()))
            clusters = build_time_clusters_for_image(
                sims[i], cap_url_ids, top_k, url2loc_text, url2date,
                gt_urls, Nmin, Nwindow
            )
            if split_name != "train" and len(clusters) < 2:
                clusters = []
            cands[img_url] = clusters
        return ds, img_infos, sims, cands


    train_ds, _, _, _ = build_pairs_and_cands("train", args.top_k, args.recompute_cache)

    _,   dev_img_infos, dev_sims, dev_cands = build_pairs_and_cands("dev", args.top_k, args.recompute_cache)
    train_ds = sample_train_balanced(train_ds, 1)
    dev_img_infos, dev_sims, dev_cands = subset_dev(dev_img_infos, dev_sims, dev_cands, 1)

    def collate_pairs(batch):
        imgs, caps, ys = zip(*batch)
        return list(imgs), list(caps), torch.stack(ys, dim=0)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=4, collate_fn=collate_pairs)
    
    # Free the bi-encoder model from memory
    del clip_retriever
    torch.cuda.empty_cache()

    # Fresh CLIP for CE
    clip_ce = CLIPModel.from_pretrained(base_ckpt).to(device)
    clip_ce.eval()

    # Build CE model
    model = build_crossenc(
        clip_ce,
        dropout=args.dropout,
        operations=args.operations.split("-")
    ).to(device)

    head_params = list(model.head.parameters())
    trainable_params= [{"params": head_params, "lr": args.lr}]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    pos_w = get_pos_weight(train_ds).to(device)
    bce   = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    #  training loop
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

        # dev evaluation
        ret = eval_time_clusters_recall_at_1(
            model, processor,
            dev_img_infos, dev_sims, cap_url_ids, dev_cands, gt_map,
            top_k=args.top_k, bs=args.bs*2, min_clusters=args.min_clusters
        )

        r1 = ret['ce_R1']
        if r1 > best_r1 and r1 > ret['baseline_R1']:
            best_r1 = r1
            suffix = f"{args.epochs}"
            if args.seed != 123:
                suffix += f"_{args.seed}"
            model_output_path = f"clip_model/cross_encoder_event_{suffix}.pt"
            torch.save(model.head.state_dict(), model_output_path)
            print(f"saved {model_output_path}")

    print("Best dev R1", best_r1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--biencoder_ckpt", required = True)
    ap.add_argument("--base_ckpt", default="openai/clip-vit-large-patch14") 
    ap.add_argument("--top_k",  type=int, default=50)
    ap.add_argument("--bs",     type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--operations", type=str, default="concatenation")
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--cache_dir", type=str, default="cache_clip_embeds")
    ap.add_argument("--recompute_cache", default=0, type=int)
    ap.add_argument("--N_min", type=int, default=3, help="Min cluster size")
    ap.add_argument("--N_window", type=int, default=7, help= "Min window so that all articles in the cluster are in a 2 N_window + 1 temporal span")
    ap.add_argument("--min_clusters",type=int, default=2, help="Minimum number of clusters in top 100 results to consider doing reranking")
    ap.add_argument("--seed", choices=[123,456,789], type=int)

    args = ap.parse_args()
    seed_everything(args.seed)
    main(args)
