import json, os, string, time
import re
import numpy as np
import requests
from tqdm import tqdm
from dateutil import parser as dateparser
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
from haversine import haversine, Unit
from scipy.optimize import linear_sum_assignment
import spacy
from dateutil import parser
from utils import *

GEONAMES_CACHE = "geonames_results.json"
CONTINENTS = {"africa","asia","europe","north america","south america","oceania","australia","antarctica"}


def _load_cache(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return {e.get("query","").lower(): e for e in data if isinstance(e, dict) and "query" in e}
            except Exception:
                return {}
    return {}


def _save_cache(cache, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(cache.values()), f, indent=2, ensure_ascii=False)


def extract_named_entities(text, model, entity_type):
    '''
    Return a list of entities of a certain type contained in a string.
    Params:
        text (str) : the text string
        model (object) : the spaCy NLP model used to process the text
        entity_type (str) : the type of entity to search for. One of ["date_and_times", "locations"]
    '''
    # Process the input text using spaCy
    doc = model(text)
    # Initialize a list to store the extracted entities
    entities = []
    current_entity = []

    # Define a mapping of entity type names to spaCy labels
    entity_type_map = {

        "dates_and_times": ["DATE", "TIME"],
        "locations": ["LOC", "GPE"]
    }
    # Iterate through the tokens in the processed text
    for token in doc:
        # Check if the token is an entity of the specified type
        if token.ent_type_ in entity_type_map[entity_type]:
            if token.ent_iob_ == 'B':  # Beginning of an entity
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                current_entity.append(token.text)
            else:  # Inside or last token of an entity
                current_entity.append(token.text)
    # Add the last entity if the sentence ends with one
    if current_entity:
        entities.append(' '.join(current_entity))     
    return entities


def get_numeric_date_label(date,spacy_model):
    '''
    Convert dates to numeric labels
    '''
    date_NER = extract_named_entities(date,spacy_model, "dates_and_times")
    output=[]
    for d in date_NER:
        try:
            output.append(parser.parse(d).replace(tzinfo=tzutc()))
        except:
            pass
    if len(output)==0:
        output='not enough information'
    else:
        output = [d.isoformat() for d in output]
    return output


# Geonames helpers
def _get_geonames_hierarchy(geoname_id, username):
    url = f"http://api.geonames.org/hierarchyJSON?geonameId={geoname_id}&username={username}"
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(e)
        print('Failed to query geonames')
        data = {'geonames':[]}

    hierarchy = [it["name"] for it in data.get("geonames", [])]
    return [h for h in hierarchy if h.lower() != "earth"]


def _search_location(name, username, max_results = 5, sleep= 3):
    url = f"http://api.geonames.org/searchJSON?q={name}&maxRows={max_results}&username={username}"
    try:
        r = requests.get(url, timeout=30); r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(e)
        print('Failed to query geonames')
        data = {'totalResultsCount':0}
    out = []
    if data.get("totalResultsCount", 0) > 0:
        for d in data.get("geonames", []):
            if "lat" in d and "lng" in d:
                out.append({
                    "query": name,
                    "geonameId": d["geonameId"],
                    "name": d.get("name"),
                    "countryName": d.get("countryName"),
                    "coordinates": (float(d["lat"]), float(d["lng"])),
                    "hierarchy": _get_geonames_hierarchy(d["geonameId"], username),
                })
                break
        time.sleep(sleep)
    return out


def _coerce_hierarchy_to_list(raw):
    if isinstance(raw, list):
        seq = raw
    elif isinstance(raw, str):
        delim = ';' if ';' in raw else ','
        seq = [t for t in (s.strip() for s in raw.split(delim)) if t]
    else:
        seq = []
    # Normalize, remove Earth, drop empties, collapse consecutive duplicates (e.g. when Paris is repeated multiple times in the same hierarchy)
    cleaned = []
    for t in seq:
        tt = (t or '').strip()
        if not tt:
            continue
        if tt.lower() == 'earth': 
            continue
        if len(cleaned) == 0 or cleaned[-1].lower() != tt.lower():
            cleaned.append(tt)
    return cleaned


def get_location_hierarchy_cached(query, username):
    norm = (query or "").split(",")[0].strip().lower()
    if not norm:
        return []
    cache = _load_cache(GEONAMES_CACHE)
    if norm in cache and "hierarchy" in cache[norm]:
        return _coerce_hierarchy_to_list(cache[norm]["hierarchy"])

    if username:
        res = _search_location(norm, username)
        if res:
            entry = res[0]
            entry["hierarchy"] = _coerce_hierarchy_to_list(entry.get("hierarchy"))
            cache[norm] = entry
            _save_cache(cache, GEONAMES_CACHE)
            return entry["hierarchy"]

    cache[norm] = {"query": norm, "hierarchy": []}
    _save_cache(cache, GEONAMES_CACHE)
    return []


def get_location_coordinates_cached(query, username):
    norm = query.split(",")[0].strip().lower()
    if not norm: return []
    cache = _load_cache(GEONAMES_CACHE)
    if norm in cache and cache[norm].get("coordinates"):
        c = cache[norm]["coordinates"]
        return [tuple(c)] if isinstance(c, (list, tuple)) else []
    if username:
        res = _search_location(norm, username)
        if res:
            cache[norm] = res[0]
            _save_cache(cache, GEONAMES_CACHE)
            c = res[0].get("coordinates")
            return [tuple(c)] if isinstance(c, (list, tuple)) else []
    cache[norm] = {"query": norm, "hierarchy": [], "coordinates": None}
    _save_cache(cache, GEONAMES_CACHE)
    return []


#  Date helpers 
def normalize_date_str(s):
    s = str(s).strip()
    if "T" in s:
        s = s.split("T", 1)[0]
    return s


def convert_to_date(s):
    if s is None:
        return None
    try: 
        return dateparser.parse(s)
    except Exception: 
        return None


def date_distance_years(dt1, dt2):
    if dt1 is None or dt2 is None:
        return float("inf")
    if dt1.tzinfo is None: 
        dt1 = dt1.replace(tzinfo=tzutc())
    if dt2.tzinfo is None: 
        dt2 = dt2.replace(tzinfo=tzutc())
    delta = relativedelta(dt1, dt2)
    return abs(delta.years + delta.months/12 + delta.days/365.25)


def norm_loc_str(s):
    _punct_trans = str.maketrans('', '', string.punctuation)
    return (s or "").lower().translate(_punct_trans).strip()


def em_pm_location_at_k(cands, gt, ks):
    scores = {}
    parts = [p.strip() for p in str(gt).split(",") if p.strip()]
    parts_norm = [norm_loc_str(p) for p in parts]
    for k in ks:
        scores[f"EM@{k}"] = 0
        scores[f"PM@{k}"] = 0
        for c in cands[:k]:
            c_norm = norm_loc_str(c)
            if parts_norm[0] in c_norm:
                #The lowest level of the ground truth location is in the predicted location
                scores[f"EM@{k}"] = 1
                scores[f"PM@{k}"] = 1
                break
            cand_head = c_norm.split(",")[0].split('(')[0]
            if any(cand_head == higher for higher in parts_norm[1:]):
                scores[f"PM@{k}"] = 1
    return scores



def em_pm_time_at_k(cands, gt, ks):
    scores = {}
    gt = gt.split('-')
    for k in ks:
        scores[f"EM@{k}"] = 0
        scores[f"PM@{k}"] = 0
        for c in cands[:k]:

            c = normalize_date_str(c)
            if len(gt)==1: 
                if 's' in gt[0].lower():
                    #ground truth is a decade
                    decade = int(gt[0][:4])
                    decade_range = [str(year) for year in range(decade,decade+10,1)]
                    if c.split('-')[0] in decade_range:
                        scores[f"EM@{k}"] = 1
                        scores[f"PM@{k}"] = 1
                        break                          
                if gt[0] in c:
                    #ground truth is a year
                    scores[f"EM@{k}"] = 1
                    scores[f"PM@{k}"] = 1
                    break 
            if len(gt)==2:
                #ground truth is year-month
                if '-'.join(gt) in c:
                    scores[f"EM@{k}"] = 1
                    scores[f"PM@{k}"] = 1
                    break
                if gt[0] in c:
                    scores[f"PM@{k}"] = 1                                     
            if len(gt)==3:          
                #ground truth is a full date:
                if '-'.join(gt)==c:
                    scores[f"EM@{k}"] = 1
                    scores[f"PM@{k}"] = 1 
                    break
                if  len(c.split('-')) > 1:
                    if gt[0]==c.split('-')[0] and  gt[1]==c.split('-')[1]:
                        scores[f"PM@{k}"] = 1 
                if gt[0]==c.split('-')[0]:  # Year matches but the rest differs
                    scores[f"PM@{k}"] = 1            
    return scores


def time_hierarchy(s):
    s = normalize_date_str(s)
    if s.endswith("s") and s[:-1].isdigit():
        decade = int(s[:-1]); cent = (decade // 100) + 1
        return [f"{decade}s", f"{cent}th century"]
    parts = s.split("-")
    out = []
    try:
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            out += [f"{y}-{m:02d}-{d:02d}", f"{y}-{m:02d}", f"{y}"]
        elif len(parts) == 2:
            y, m = int(parts[0]), int(parts[1]); out += [f"{y}-{m:02d}", f"{y}"]
        else:
            y = int(parts[0]); out += [f"{y}"]
        decade = (y // 10) * 10
        out.append(f"{decade}s")
        out.append(f"{((y-1)//100)+1}th century")
    except Exception:
        out = [s]
    return out


def f1_from_sets(a, b):
    if not a and not b: 
        return 1.0
    if not a or not b: 
        return 0.0
    inter = len(a & b)
    return 2 * inter / (len(a) + len(b))


def _cumulative_paths_from_hierarchy(hier_nodes):
    """
    Build hierarchical label set as suffixes of the full leaf→root chain.
    Example input nodes: ["Asia", "Japan", "Tokyo"]
    Output set: {"tokyo, japan, asia", "japan, asia", "asia"}
    """
    if not hier_nodes:
        return set()
    # leaf→root order
    rev = list(reversed(hier_nodes))
    def dedup_seq(seq):
        out = []
        for x in seq:
            if not out or out[-1].lower() != (x or "").lower():
                out.append(x)
        return out
    rev = dedup_seq(rev)  # ensure unique consecutive items in the chain
    paths = set()
    for i in range(len(rev)):
        sub = dedup_seq(rev[i:])
        path = ", ".join(sub).lower().strip()
        if path:
            paths.add(path)
    return paths


def example_f1_location(gt, pred, geonames_user):
    gt_nodes = get_location_hierarchy_cached(gt, geonames_user)
    pr_nodes = get_location_hierarchy_cached(pred, geonames_user)

    gt_set = _cumulative_paths_from_hierarchy(gt_nodes)
    pr_set = _cumulative_paths_from_hierarchy(pr_nodes)
    
    if not gt_set and not pr_set:
        return 1.0
    if not gt_set or not pr_set:
        return 0.0
    inter = len(gt_set & pr_set)
    return 2 * inter / (len(gt_set) + len(pr_set))


def example_f1_time(gt, pred):
    return f1_from_sets(set(time_hierarchy(gt)), set(time_hierarchy(pred)))


def codelta_score(pred_coords, gt_coords):
    if not pred_coords or not gt_coords: 
        return 0.0
    M = np.array([[haversine(p, g, unit=Unit.KILOMETERS) for g in gt_coords] for p in pred_coords])
    r, c = linear_sum_assignment(M)
    vals = sorted([M[i, j] for i, j in zip(r, c)])
    return float(sum(1/(1+v) for v in vals) / len(gt_coords))


#CO delta
def codelta_from_strings(pred_loc, gt_loc, geonames_user):
    pred_coords = get_location_coordinates_cached(pred_loc, geonames_user)
    gt_coords = get_location_coordinates_cached(gt_loc, geonames_user)
    return codelta_score(pred_coords, gt_coords)


#  GT loading 
def load_gt(dataset, split, filter_ris):
    dataset_paths = {
        "tara": {
            "dev":  "data/TARA_dataset/input/gold_dev.jsonl",
            "test": "data/TARA_dataset/input/gold_test.jsonl",
            "interest": "data/TARA_dataset/input/gold_interest.jsonl",
        },
        "5pils_ooc": {
            "test": "data/5pils_ooc/test.json",
        },
    }

    path = dataset_paths[dataset][split]

    if dataset == "tara":
        data = load_jsonl(path)
    else:
        data = load_json(path)
        if filter_ris=='ris':
            data = [d for d in data if len(d['evidence_captions']) > 0]
        if filter_ris =='non_ris':
            data = [d for d in data if len(d['evidence_captions'])== 0]

    out = {}
    for idx in range(len(data)):
        # image_path resolution
        if dataset=='tara':
            image_path =  f"data/TARA_dataset/img/{split}/{idx}.png"
            loc = data[idx]['gold_location']
            dt = data[idx]['gold_time']
        else:
            image_path = f"data/5pils_ooc/{data[idx]['image_path']}"
            loc = data[idx]['location']
            loc = loc if loc!= 'not enough information' else None
            dt = data[idx]['date_numeric_label']
            dt = dt[0].split('T')[0] if dt!= 'not enough information' else None

        out[image_path] = {"location": loc, "date": dt}

    return out


#  Candidate builders-
def get_candidates(urls, task, corpus, 
                          web_urls_to_articles):
    out = []
    for u in urls:
        if u in web_urls_to_articles.keys():
            meta = corpus[web_urls_to_articles[u]]
            gl = [k['value'] for k in meta['keywords'] if k['name']=='glocations']
            dt = meta.get("pub_date")
            if task == "location":

                if gl:
                    cand_loc = ", ".join(gl)
                else:
                    cand_loc = "" # no location
                if cand_loc:
                    out.append(cand_loc)
                continue

            else:
                if not dt:
                    continue
                pred_date = normalize_date_str(dt)

                out.append(pred_date)
    return out


# GREAT
DMAX_KM = 1000.0  # normalization for geo
# tolerances and weights for temporal scoring
_TOL = dict(decade=50, year=5, month=6, day=15)
_W   = dict(century=1.0, decade=1.0, year=1.25, month=1.5, day=1.5)

def _time_units(date_str):
    """Return {'century','decade','year','month','day'} as ints or None."""
    s = normalize_date_str(str(date_str))
    if not s:
        return dict(century=None, decade=None, year=None, month=None, day=None)

    # decade like "1970s"
    m = re.match(r"^(\d{4})s$", s)
    if m:
        y = int(m.group(1))
        return dict(
            century=((y - 1) // 100) + 1,
            decade=(y // 10) * 10,
            year=None, month=None, day=None
        )

    parts = s.split("-")
    try:
        y = int(parts[0])
        century = ((y - 1) // 100) + 1
        decade = (y // 10) * 10
        year = y
        month = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else None
        day = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
        return dict(century=century, decade=decade, year=year, month=month, day=day)
    except Exception:
        return dict(century=None, decade=None, year=None, month=None, day=None)
    

def _unit_score(gu, pu, unit):
    if gu is None or pu is None:
        return 0.0
    if unit == "century":
        return 1.0 if gu == pu else 0.0
    T = _TOL[unit]
    return max(0.0, 1.0 - (abs(gu - pu) / float(T)))


def great_temp_score(gt_date, pred_date):
    g = _time_units(gt_date)
    p = _time_units(pred_date)
    num = 0.0
    den = 0.0
    for unit in ["century", "decade", "year", "month", "day"]:
        w = _W[unit]
        num += w * _unit_score(g[unit], p[unit], unit)
        den += w
    return num / den if den > 0 else 0.0


def great_geo_score(gt_loc, pred_loc, geonames_user):
    gt_c = get_location_coordinates_cached(gt_loc, geonames_user)
    pr_c = get_location_coordinates_cached(pred_loc, geonames_user)
    if not gt_c:
        return 0.0
    if not pr_c:
        return 0.0
    d_km = haversine(tuple(gt_c[0]), tuple(pr_c[0]), unit=Unit.KILOMETERS)
    return max(0.0, 1.0 - (d_km / DMAX_KM))


# Evaluation loop
def evaluate(
    results_file,
    dataset,
    split,
    task,
    ks,
    geonames_user,
    filter_ris = False,  #if true, only take non RIS results for evaluation
    ):
    # load corpus before any evaluation
    nlp = spacy.load('en_core_web_lg')
    articles = []
    for f in sorted(os.listdir("processed_articles")):
        if dataset=='tara':
            #Only remove 2022, 2023, and Guardian part 2 if it is TARA
            if ('2022' in f) or ('2023' in f) or ('guardian_part2' in f) :
                continue
        if f.endswith(".json"):
            articles += load_json(os.path.join("processed_articles", f))
    # qwen2.5 category 1 filter
    qwen_class = []
    for f in sorted(os.listdir("qwen2.5_article_class")):
        if dataset=='tara':
            if ('2022' in f) or ('2023' in f) or ('guardian_part2' in f):
                continue
        qwen_class += load_json(os.path.join("qwen2.5_article_class", f))
    qwen_class = {q['web_url']:q['output'][0]  for q in qwen_class}
    # keep only category 1
    articles = [articles[a] for a in range(len(articles)) if qwen_class[articles[a]['web_url']].lower()=='category 1']
    #Remove all articles that do not have a keyword location
    articles = [a for a in articles if 'glocations' in [kw['name'].lower() for kw in a['keywords']]]
    articles += load_json('gt_articles/train.json')
    web_urls_to_articles = {articles[a]['web_url']:a for a in range(len(articles))}
    # load GT
    gt_map = load_gt(dataset, split, filter_ris)

    # load results
    results_obj = None

    results_obj = load_json(results_file)

    # accumulators
    em_at = {k: [] for k in ks}
    pm_at = {k: [] for k in ks}
    ex_f1 = []
    if task == "location":
        co_list, great_geo_list = [],  []
    else:
        delta_list, year_dist_list, great_time_list = [], [], []


    iterator = [{"image_path": ip, "list": lst} for ip, lst in results_obj.items()]
    for it in tqdm(iterator, desc="Evaluating"):
        img = it.get("image_path")
        if not img or img not in gt_map:
            continue
        gt = gt_map[img]["location" if task == "location" else "date"]
        if not gt: 
            continue
        cands = get_candidates(
            it["list"], task, articles,
            web_urls_to_articles=web_urls_to_articles
        )

        cands = [normalize_date_str(c) if task == "time" else c for c in cands if c and str(c).strip()]

        if not cands:
            for k in ks:
                em_at[k].append(0)
                pm_at[k].append(0)
            if task == "location":
                ex_f1.append(0)
                co= 0.0
                co_list.append(co)
                great_geo_list.append(0)
            else:
                ex_f1.append(0)
                delta_list.append(0.0)
                year_dist_list.append(0.0)
                great_time_list.append(0)
            continue

        # EM and PM at K
        if task == "location":
            scores = em_pm_location_at_k(cands, gt, ks)
        else:
            scores = em_pm_time_at_k(cands, gt, ks)
        for k in ks:
            em_at[k].append(scores[f"EM@{k}"])
            pm_at[k].append(scores[f"PM@{k}"])
        # Extra metrics from top 1
        top1 = cands[0]
        if task == "location":
            if scores["EM@1"] == 1:
                ex_f1.append(1.0)
                co_list.append(1.0)
                great_geo_list.append(1.0)
            else:
                great_geo_list.append(great_geo_score(gt, top1, geonames_user))
                ex_f1.append(example_f1_location(gt, top1, geonames_user))
                co = codelta_from_strings(top1, gt, geonames_user)
                co_list.append(co)
        else:
            if scores["EM@1"] == 1:
                ex_f1.append(1.0)
                delta_list.append(1.0)
                year_dist_list.append(0.0)
                great_time_list.append(1.0)
            else:
                ex_f1.append(example_f1_time(gt, top1))
                gt_dt = convert_to_date(normalize_date_str(gt))
                pr_dt = convert_to_date(normalize_date_str(top1))
                yd = date_distance_years(pr_dt, gt_dt)
                delta = 1/(1+yd) if np.isfinite(yd) else 0.0
                delta_list.append(delta)
                year_dist_list.append(yd if np.isfinite(yd) else 0.0)
                great_time_list.append(great_temp_score(gt, top1))

    # report
    n = len(next(iter(em_at.values()))) if em_at else 0
    print(f"Instances evaluated: {n}")
    for k in ks:
        em = round(100 * sum(em_at[k]) / max(1, len(em_at[k])), 2)
        pm = round(100 * sum(pm_at[k]) / max(1, len(pm_at[k])), 2)
        print(f"EM@{k}: {em} | PM@{k}: {pm}")
    print(f"Example-F1: {round(100 * (sum(ex_f1) / max(1, len(ex_f1))), 2)}")
    if task == "location":
        print(f"codelta: {round(100* sum(co_list) / max(1, len(co_list)), 4)}")
        print(len(great_geo_list))
        print(f"GREAT_geo: {round(100 * (sum(great_geo_list) / max(1, len(great_geo_list))), 2)}")
        return round(100 * (sum(great_geo_list) / max(1, len(great_geo_list))))
    else:
        print(f"delta: {round(100 * (sum(delta_list) / max(1, len(delta_list))), 2)}")
        print(len(great_time_list))
        print(f"GREAT_time: {round(100 * (sum(great_time_list) / max(1, len(great_time_list))), 2)}")

        return round(100 * (sum(great_time_list) / max(1, len(great_time_list))))