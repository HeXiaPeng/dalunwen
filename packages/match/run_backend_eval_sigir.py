import os
import re
import json
import requests

BASE_BACKEND = "http://localhost:3001"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset", "sigir")

AGE_RE = re.compile(r"(\d+)\s*[-]?\s*year[- ]?old", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(male|female)\b", re.IGNORECASE)

def extract_profile(text: str):
    age_match = AGE_RE.search(text or "")
    age = int(age_match.group(1)) if age_match else 45
    gender_match = GENDER_RE.search(text or "")
    gender = gender_match.group(1).capitalize() if gender_match else "Female"
    return age, gender

def load_queries():
    path = os.path.join(DATA_DIR, "queries.jsonl")
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            queries.append(obj)
    return queries

def load_qrels():
    path = os.path.join(DATA_DIR, "qrels", "test.tsv")
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            parts = line.strip().split()
            if first and parts and "query-id" in parts[0]:
                first = False
                continue
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
                try:
                    rel = int(rel)
                except:
                    rel = 1
                if rel > 0:
                    gold.setdefault(qid, set()).add(docid)
    return gold

def register_and_login():
    # register
    sess = requests.Session()
    r = sess.post(f"{BASE_BACKEND}/api/users/register", json={"username": "sigir_eval", "password": "sigir_eval_pwd"})
    if r.status_code == 200 and r.json().get("code") == 200:
        token = r.json()["data"]["token"]
    else:
        # login
        r = sess.post(f"{BASE_BACKEND}/api/users/login", json={"username": "sigir_eval", "password": "sigir_eval_pwd"})
        r.raise_for_status()
        token = r.json()["data"]["token"]
    return token

def eval_small_sample(k=10, sample_count=5):
    token = register_and_login()
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    queries = load_queries()[:sample_count]
    gold = load_qrels()
    total_prec = 0.0
    total_hit = 0
    details = []
    for q in queries:
        qid = q.get("_id")
        text = q.get("text") or ""
        age, gender = extract_profile(text)
        payload = {
            "age": age,
            "gender": gender,
            "condition": text[:800],
            "isRecruiting": True,
            "registry": "usa"
        }
        r = requests.post(f"{BASE_BACKEND}/api/match/patient", headers=headers, json=payload)
        if r.status_code != 200:
            details.append({"qid": qid, "error": r.text})
            continue
        data = r.json().get("data", [])
        top_ids = [item.get("trial_id") for item in data[:k] if item.get("trial_id")]
        gset = gold.get(qid, set())
        hits = sum(1 for tid in top_ids if tid in gset)
        prec = hits / k if k else 0.0
        hit = 1 if hits > 0 else 0
        total_prec += prec
        total_hit += hit
        details.append({"qid": qid, "p_at_k": round(prec,3), "hit": hit, "gold_count": len(gset), "top_ids": top_ids})
    avg_p = total_prec / len(queries) if queries else 0.0
    hit_rate = total_hit / len(queries) if queries else 0.0
    result = {"k": k, "samples": len(queries), "avg_p_at_k": round(avg_p,3), "hit_rate_at_k": round(hit_rate,3), "details": details}
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    eval_small_sample(k=10, sample_count=5)
