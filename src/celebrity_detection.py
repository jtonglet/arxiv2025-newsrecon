import os
import json
import time
import boto3
from pathlib import Path
from botocore.config import Config
from botocore.exceptions import ClientError
from utils import *

def build_split_infos(dataset, 
                      split
                      ):
    """
    Returns list of (img_path, img_url) for the chosen split.
    """
    if dataset == 'tara':
        split_file = f"data/TARA_dataset/input/gold_{split}.jsonl"
        entries = load_jsonl(split_file)
        for i, x in enumerate(entries):
            x["orig_idx"] = i
        img_infos = [(f"data/TARA_dataset/img/{split}/{x['orig_idx']}.png", x["web_url"]) for x in entries]
    else:
        # 5pils-OOC
        split_file = 'data/5pils_ooc/test.json'
        entries = load_json(split_file)[::2]
        img_infos = [(f"data/5pils_ooc/{d['image_path']}", d['URL']) for d in entries]
    return img_infos


def create_client():
    """
    Create AWS client session for the celebrity rekognition API
    """
    aws_key = os.getenv("YOUR_AWS_ACCESS_KEY")
    aws_secret = os.getenv("YOUR_AWS_SECRET_KEY")
    aws_region = "YOUR_AWS_REGION"

    if not (aws_key and aws_secret and aws_region):
        raise ValueError("Missing AWS credentials in environment variables!")

    session = boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
        region_name=aws_region
    )
    cfg = Config(retries={"max_attempts": 10, "mode": "adaptive"})
    return session.client("rekognition", config=cfg)


def recognize_celebrities(client,
                          image_path, 
                          max_retries=6):
    """
    Detects celebrities in an image.
    """
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    SLEEP= 2
    for _ in range(max_retries):
        try:
            resp = client.recognize_celebrities(Image={"Bytes": img_bytes})
            celebs = []
            for c in resp.get("CelebrityFaces", []):
                celebs.append({
                    "Name": c.get("Name"),
                    "Id": c.get("Id"),
                    "MatchConfidence": float(c.get("MatchConfidence", 0.0)),
                    "Urls": c.get("Urls", []),
                    "Face": {
                        "BoundingBox": c.get("Face", {}).get("BoundingBox"),
                        "Confidence": float(c.get("Face", {}).get("Confidence", 0.0)),
                    }
                })
            return celebs
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in {"ThrottlingException", "ProvisionedThroughputExceededException", "TooManyRequestsException"}:
                time.sleep(SLEEP)
                continue
            # non retryable client error
            return {"error": f"{code}: {e.response.get('Error', {}).get('Message', str(e))}"}
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Failed after {max_retries} retries"}


def get_celebrities_dataset(
                dataset_key, 
                split, 
                out_json_path):
    """
    Main function to get the celbrities for all instances in a dataset split.
    """
    #Prepare the dataset
    infos = build_split_infos(dataset_key, split, n_instances=1644)

    #Create the client
    client = create_client()

    results = {}
    for idx, (img_path, _) in enumerate(infos, start=1):
        p = Path(img_path)
        if not p.exists():
            results[img_path] = {"error": "file not found"}
            continue
        payload = recognize_celebrities(client, img_path)
        results[img_path] = payload

    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)



if __name__=="__main__":
    get_celebrities_dataset("tara", "test", "rekog_celeb_tara_test.json")
    get_celebrities_dataset("5pils", "test", "rekog_celeb_5pils_ooc.json")