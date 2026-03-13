import os
import json
import re
import math
import requests
import argparse
from collections import Counter, defaultdict
import csv

BASE_BACKEND = "http://localhost:3001"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset", "sigir")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_corpus():
    corpus = []
    for doc in load_jsonl(os.path.join(DATA_DIR, "corpus.jsonl")):
        corpus.append(doc)
    return corpus

def load_queries():
    return list(load_jsonl(os.path.join(DATA_DIR, "queries.jsonl")))

def load_qrels():
    gold = defaultdict(set)
    path = os.path.join(DATA_DIR, "qrels", "test.tsv")
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
                    gold[str(qid)].add(str(docid))
    return gold

class SimpleBM25:
    def __init__(self, docs):
        self.docs = []
        self.df = defaultdict(int)
        self.avgdl = 0.0
        for doc in docs:
            tf = Counter(self.tokenize(doc))
            self.docs.append({"tf": tf, "len": sum(tf.values()), "id": doc.get("_id")})
            for t in tf.keys():
                self.df[t] += 1
        self.avgdl = sum(d["len"] for d in self.docs)/len(self.docs) if self.docs else 0.0
    @staticmethod
    def tokenize(doc):
        text = " ".join([doc.get("title",""), doc.get("text","")]).lower()
        tokens = []
        word=[]
        for ch in text:
            if ch.isalnum():
                word.append(ch)
            else:
                if word:
                    tokens.append("".join(word))
                    word=[]
        if word:
            tokens.append("".join(word))
        return tokens
    @staticmethod
    def tokenize_query(text):
        text = (text or "").lower()
        tokens=[]
        word=[]
        for ch in text:
            if ch.isalnum():
                word.append(ch)
            else:
                if word:
                    tokens.append("".join(word))
                    word=[]
        if word:
            tokens.append("".join(word))
        return tokens
    def search(self, query, top_k=50, k1=1.2, b=0.75):
        qtf = Counter(self.tokenize_query(query))
        N = len(self.docs)
        scores=[]
        for d in self.docs:
            s=0.0
            dl = d["len"] or 1
            for term in qtf:
                df = self.df.get(term,0)
                if df==0:
                    continue
                idf = math.log((N-df+0.5)/(df+0.5)+1)
                f = d["tf"].get(term,0)
                if f==0:
                    continue
                denom = f + k1*(1-b + b*dl/(self.avgdl or 1))
                s += idf*(f*(k1+1))/denom
            scores.append((d["id"], s))
        scores.sort(key=lambda x:x[1], reverse=True)
        return scores[:top_k]

def register_and_login():
    sess = requests.Session()
    r = sess.post(f"{BASE_BACKEND}/api/users/register", json={"username":"trialgpt_small","password":"trialgpt_small_pwd"})
    if r.status_code == 200 and r.json().get("code")==200:
        token = r.json()["data"]["token"]
    else:
        r = sess.post(f"{BASE_BACKEND}/api/users/login", json={"username":"trialgpt_small","password":"trialgpt_small_pwd"})
        r.raise_for_status()
        token = r.json()["data"]["token"]
    return token

def build_prompt(patient_text, candidates):
    lines=[]
    for doc in candidates:
        cid = doc["_id"]
        title = doc.get("title","")
        text = (doc.get("text","") or "")
        text = text.replace("\n"," ").strip()[:300]
        lines.append(f"ID: {cid}, Title: {title}, Text: {text}")
    prompt = f"""
You are a clinical trial matching assistant. Given a patient case and a list of candidate clinical trials (ID, Title, Text), select the top 10 most suitable trials and assign each a score (0-100). Consider eligibility and relevance broadly.

Patient:
{patient_text[:800]}

Candidates:
{os.linesep.join(lines)}

Return STRICT JSON only:
{{"trials":[{{"trial_id":"...", "score":90, "reason":"..."}}]}}
"""
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen-max")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--export", type=str, default="results_sigir_llm_small.csv")
    args = parser.parse_args()
    token = register_and_login()
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    corpus = load_corpus()
    id2doc = {d["_id"]: d for d in corpus}
    bm25 = SimpleBM25(corpus)
    queries = load_queries()[:args.samples]
    gold = load_qrels()
    k=10
    total_prec=0.0
    total_hit=0
    results=[]
    for q in queries:
        qid = q["_id"]
        text = q["text"]
        top_ids = [docid for docid,_ in bm25.search(text, top_k=50)]
        candidates = [id2doc[i] for i in top_ids if i in id2doc]
        prompt = build_prompt(text, candidates)
        payload = {"model": args.model, "messages":[{"role":"system","content":"You output only valid JSON."},{"role":"user","content":prompt}]}
        r = requests.post(f"{BASE_BACKEND}/api/ai/generate", headers=headers, json=payload)
        if r.status_code!=200:
            results.append({"qid":qid,"error":r.text})
            continue
        content = r.json().get("data","")
        content = str(content).replace("```json","").replace("```","").strip()
        trials=[]
        try:
            parsed = json.loads(content)
            trials = parsed["trials"] if isinstance(parsed, dict) else parsed
        except:
            trials=[]
        pred_ids = [t.get("trial_id") for t in trials[:k] if t.get("trial_id")]
        gset = gold.get(qid,set())
        hits = sum(1 for tid in pred_ids if tid in gset)
        prec = hits/k if k else 0.0
        hit = 1 if hits>0 else 0
        total_prec += prec
        total_hit += hit
        results.append({"qid":qid,"p_at_k":round(prec,3),"hit":hit,"gold_count":len(gset),"pred_ids":pred_ids})
    avg_p = total_prec/len(queries) if queries else 0.0
    hit_rate = total_hit/len(queries) if queries else 0.0
    summary = {"avg_p_at_k":round(avg_p,3),"hit_rate_at_k":round(hit_rate,3),"details":results}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # export CSV
    out_path = os.path.join(os.path.dirname(__file__), args.export)
    if results:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

if __name__=="__main__":
    main()
