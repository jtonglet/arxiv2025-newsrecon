from PIL import Image
import json
import re
import os 
import random
import numpy as np
import hashlib
import pathlib
from torchvision.transforms.functional import InterpolationMode
import torch
import torchvision.transforms as T
import torch, torch.nn.functional as F
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def seed_everything(seed = 123):
    #set random seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean(s):
    #Helper function to clean a string from unnecessary white spaces
    return " ".join(str(s).split()).strip()


def load_jsonl(file_path):
    '''
    Load a jsonl file
    '''
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path):
    '''
    Load json file
    '''
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def save_json(obj, p):
    '''
    Save json file
    '''
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def is_valid_loc(geolocs, available_geolocs):
    '''
    Verify that a location keyword is valid
    '''
    standard_mapping = {
        'united states': 'usa',
        'united states of america': 'usa',
        'united kingdom': 'uk',
        'scotland': 'uk',
        'wales': 'uk',
        'england': 'uk',
        'northern ireland': 'uk',
        'united arab emirates': 'uae'
    }
    geolocs = [standard_mapping[g]  if g in standard_mapping.keys() else g for g in geolocs]
    available_geolocs = [standard_mapping[g] if g in standard_mapping.keys() else g for g in available_geolocs ]
    for g in geolocs:
       for ag in available_geolocs:
            if ag in g:
                return True
    return False


def norm_name(s):
    #Helper function to normalize strings of locations
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_loc_atoms(loc_text):
    '''
    Extract atomic location keywords from the article metadata
    '''
    atoms = set()
    for seg in loc_text.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        base = re.sub(r"\(.*?\)", "", seg).strip()
        if base:
            atoms.add(norm_name(base))
        for m in re.findall(r"\((.*?)\)", seg):
            m = m.strip()
            if m:
                atoms.add(norm_name(m))
    return atoms


def build_url2loc_text_from_articles(articles_list, allowed_urls):
    #Helper function to map article URLS to their locations
    url2loc = {}
    for a in articles_list:
        url = a.get("web_url")
        if not url or url not in allowed_urls:
            continue
        kws = a.get("keywords", []) or []
        locs = [clean(k.get("value", "")) for k in kws
                if clean(k.get("name", "")).lower() == "glocations" and k.get("value")]
        locs = [x for x in dict.fromkeys(locs) if x]
        if locs:
            url2loc[url] = " ; ".join(locs)
    return url2loc


def build_url2cap_first(captions_list):
    #Helper function to make article URL to the generated captions
    url2cap = {}
    for c in captions_list:
        url = c.get("web_url")
        outs = c.get("output") or []
        if url and outs and outs[0].strip():
            url2cap.setdefault(url, outs[0])
    return url2cap


def try_load_feats(cache_dir, key):
    npz_path, meta_path = cache_paths(cache_dir, key)
    if not npz_path.exists() or not meta_path.exists():
        return None, None
    data = np.load(npz_path)
    feats = data["feats"]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return torch.from_numpy(feats), meta["ids"]


def hash_list_str(items):
    h = hashlib.sha1()
    for it in items:
        h.update((it if isinstance(it, str) else str(it)).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def cache_key(model_id, ckpt_name, tag, ids_ordered):
    ids_for_hash = [x if isinstance(x, str) else (x[0] if isinstance(x, tuple) else str(x)) for x in ids_ordered]
    return f"{tag}__{hashlib.sha1((model_id + '||' + ckpt_name).encode()).hexdigest()}__{hash_list_str(ids_for_hash)}"


def cache_paths(cache_dir, key):
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"{key}.npz"
    meta = cache_dir / f"{key}.meta.json"
    return npz, meta


def save_feats(cache_dir, key, feats_np, ids_ordered, extra_meta=None):
    npz_path, meta_path = cache_paths(cache_dir, key)
    np.savez_compressed(npz_path, feats=feats_np)
    meta = {"ids": ids_ordered, "extra": extra_meta or {}}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return str(npz_path)


def extract_first_list(text):
    """
    Extract the first list that appears as text within a given string and convert it to a Python list.
    This is used in caption generation, in the case that the Qwen outputs more than one list of captions.
    """
    # Regular expression to find a list in the format [item1, item2, ...]
    delimiters = ["', '", '", "', '\', "', '", \'', ', \\']
    pattern = '|'.join(map(re.escape, delimiters))
    text =text.replace('[Name]', '').replace('[name]', '')
    text = text.split("]")[0].split("\n")[0]

    l = [cap.replace("'","").replace("[","").replace("]", "") for cap in re.split(pattern, text)]
    return l


def get_article_content(data, source='nyt'):
    # Get the content of an article
    if source=='nyt':
        article = data['abstract'] + '\n'
        if data['abstract']!= data['snippet']:
            article += data['snippet'] + '\n'
        if data['abstract']!=data['lead_paragraph']:
            article += data['lead_paragraph']
    else:
        article = data['fields']['headline'] + '\n'
        if 'trailText' in data['fields'].keys():
            article += data['fields']['trailText'] + '\n'
        if 'body' in data['fields'].keys():
            article += data['fields']['body'] + '\n'
    return article


def get_time_hierarchy(date_str, include_full_date=True, include_year_month=True):
    """
    Convert the date string into a hierarchy list for the Example-F1 metric.
    
    Parameters:
    date_str (str): A date string in 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD' format.
    include_full_date (bool): Whether to include the full date in the hierarchy.
    include_year_month (bool): Whether to include the year-month in the hierarchy.
    """
    # Match year, year-month, or year-month-day
    date_pattern = r'\b(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?\b'
    match = re.search(date_pattern, date_str.replace('s', ''))
    if not match:
        print(date_str)
        print("Invalid date format. Expected 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'.")
        return None
    
    # Extract components from the regex match
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else None
    day = int(match.group(3)) if match.group(3) else None
    
    # Initialize the hierarchy
    hierarchy = []
    
    # Full date
    if day and include_full_date:
        hierarchy.append(f"{year}-{month}-{day}")
    
    # Year-month
    if month and include_year_month:
        hierarchy.append(f"{year}-{month}")
    
    # Year
    hierarchy.append(str(year))
    
    # Decade
    hierarchy.append(f"{year // 10 * 10}s")
    
    # Century
    hierarchy.append(f"{(year - 1) // 100 + 1}th century")

    if 's' in date_str:
        hierarchy= [None, None, None] + hierarchy[-2:]
    
    return hierarchy


# encoders
@torch.no_grad()
def encode_caps_for_urls(model, urls, url2cap_map, processor, bs=256):
    feats, ids = [], []
    for s in tqdm(range(0, len(urls), bs), desc="encode captions"):
        texts = [url2cap_map[u] for u in urls[s:s+bs]]
        batch = processor(text=texts, return_tensors="pt",
                          padding=True, truncation=True).to(device)
        emb = model.get_text_features(**batch).float()
        feats.append(emb.cpu())
        ids.extend(urls[s:s+bs])
    feats = F.normalize(torch.cat(feats), dim=-1)
    return feats, ids


@torch.no_grad()
def encode_imgs_oneclip(model, img_infos, processor, bs=128):
    feats, ids = [], []
    for s in tqdm(range(0, len(img_infos), bs), desc="encode images"):
        paths, urls = zip(*img_infos[s:s+bs])
        imgs = [Image.open(p).convert("RGB") for p in paths]
        batch = processor(images=imgs, return_tensors="pt").to(device)
        emb   = model.get_image_features(**batch).float()
        feats.append(emb.cpu())
        ids.extend(urls)
    feats = F.normalize(torch.cat(feats), dim=-1)
    return feats, ids


@torch.no_grad()
def cached_encode_caps_for_urls(clip_model, urls, url2cap_map, processor, bs, cache_dir, ckpt_name, model_id):
    key = cache_key(model_id, ckpt_name, "caps", urls)
    feats_loaded, ids_loaded = try_load_feats(cache_dir, key)
    if feats_loaded is not None:
        feats = F.normalize(feats_loaded.float(), dim=-1)
        return feats, ids_loaded
    feats, ids = encode_caps_for_urls(clip_model, urls, url2cap_map, processor, bs=bs)
    feats_np = feats.numpy()
    save_feats(cache_dir, key, feats_np, ids, extra_meta={"type": "caps"})
    return feats, ids


@torch.no_grad()
def cached_encode_imgs(clip_model, img_infos, processor, bs, cache_dir, ckpt_name, model_id, tag, recompute_cache):
    ids_for_hash = [(p, u) for (p, u) in img_infos]
    key = cache_key(model_id, ckpt_name, tag, ids_for_hash)
    if not recompute_cache:
        feats_loaded, ids_loaded = try_load_feats(cache_dir, key)
        if feats_loaded is not None:
            feats = F.normalize(feats_loaded.float(), dim=-1)
            return feats, ids_loaded
    feats, ids = encode_imgs_oneclip(clip_model, img_infos, processor, bs=bs)
    feats_np = feats.numpy()
    save_feats(cache_dir, key, feats_np, ids, extra_meta={"type": tag})
    return feats, ids


def get_pos_weight(ds):
    base = ds.dataset.items if isinstance(ds, torch.utils.data.Subset) else ds.items
    idxs = ds.indices if isinstance(ds, torch.utils.data.Subset) else range(len(ds))
    n_pos = sum(int(base[i][2]) for i in idxs)
    n_neg = len(list(idxs)) - n_pos
    w = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    return torch.tensor([w], dtype=torch.float32)


# Image processing functions for InternVL models
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_internvl3(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values