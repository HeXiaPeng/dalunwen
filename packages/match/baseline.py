import os
import json
import math
import requests
import sys
from collections import defaultdict, Counter

BASE_BACKEND = "http://localhost:3001"
ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data", "TrialGPT", "dataset")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_corpus(ds):
    p = os.path.join(DATA_DIR, ds, "corpus.jsonl")
    corpus = list(load_jsonl(p))
    id2doc = {d["_id"]: d for d in corpus}
    return corpus, id2doc

def load_queries(ds, limit=5):
    p = os.path.join(DATA_DIR, ds, "queries.jsonl")
    qs = list(load_jsonl(p))
    return qs[:limit]

def load_qrels(ds):
    g = defaultdict(dict)
    p = os.path.join(DATA_DIR, ds, "qrels", "test.tsv")
    with open(p, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            parts = line.strip().split()
            if first and parts and "query-id" in parts[0].lower():
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
                g[str(qid)][str(docid)] = rel
    return g

class SimpleBM25:
    def __init__(self, docs):
        self.docs = []
        self.df = defaultdict(int)
        self.avgdl = 0.0
        for doc in docs:
            tf = Counter(self.tokenize(" ".join([doc.get("title",""), doc.get("text","")]).lower()))
            self.docs.append({"tf": tf, "len": sum(tf.values()), "id": doc.get("_id")})
            for t in tf.keys():
                self.df[t] += 1
        self.avgdl = sum(d["len"] for d in self.docs)/len(self.docs) if self.docs else 0.0
    @staticmethod
    def tokenize(text):
        tokens=[]; word=[]
        for ch in text:
            if ch.isalnum(): word.append(ch)
            else:
                if word:
                    tokens.append("".join(word)); word=[]
        if word: tokens.append("".join(word))
        return tokens
    def search(self, query, top_k=2000, k1=1.2, b=0.75):
        qtf = Counter(self.tokenize((query or "").lower()))
        N = len(self.docs)
        scores=[]
        for d in self.docs:
            s=0.0; dl = d["len"] or 1
            for term in qtf:
                df = self.df.get(term,0)
                if df==0: continue
                idf = math.log((N-df+0.5)/(df+0.5)+1)
                f = d["tf"].get(term,0)
                if f==0: continue
                denom = f + k1*(1-b + b*dl/(self.avgdl or 1))
                s += idf*(f*(k1+1))/denom
            scores.append((d["id"], s))
        scores.sort(key=lambda x:x[1], reverse=True)
        return [docid for docid,_ in scores[:top_k]]

def extract_criteria(text, max_inc=16, max_exc=16):
    t = (text or "")
    low = t.lower()
    inc_idx = low.find("inclusion criteria")
    exc_idx = low.find("exclusion criteria")
    inc_block = t[inc_idx:(exc_idx if exc_idx!=-1 else len(t))] if inc_idx!=-1 else ""
    exc_block = t[exc_idx:] if exc_idx!=-1 else ""
    inc_lines = [l.strip() for l in inc_block.split("\n") if l.strip()]
    exc_lines = [l.strip() for l in exc_block.split("\n") if l.strip()]
    return inc_lines[:max_inc], exc_lines[:max_exc]

def register_and_login():
    sess = requests.Session()
    r = sess.post(f"{BASE_BACKEND}/api/users/register", json={"username":"baseline_run","password":"baseline_run_pwd"})
    if r.status_code==200 and r.json().get("code")==200:
        token = r.json()["data"]["token"]
    else:
        r = sess.post(f"{BASE_BACKEND}/api/users/login", json={"username":"baseline_run","password":"baseline_run_pwd"})
        r.raise_for_status()
        token = r.json()["data"]["token"]
    return token

def call_qwen_criterion(token, model, patient_text, trial_id, crits):
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    prompt = {"patient": patient_text[:1200], "trial_id": trial_id, "criteria": crits}
    sys = "You are a clinical trial matching assistant. For each criterion, return label include|exclude|not_relevant and evidence. Return STRICT JSON."
    payload = {"model": model, "messages":[{"role":"system","content":sys},{"role":"user","content":json.dumps(prompt)}]}
    r = requests.post(f"{BASE_BACKEND}/api/ai/generate", headers=headers, json=payload)
    if r.status_code!=200:
        return None
    content = str(r.json().get("data","")).replace("```json","").replace("```","").strip()
    try:
        return json.loads(content)
    except:
        return None

def aggregate_linear(match_json):
    inc=0; exc=0; rel=0
    for c in match_json.get("criteria",[]):
        lab = (c.get("label") or "").lower()
        if lab=="include": inc+=1; rel+=1
        elif lab=="exclude": exc+=1
        elif lab=="not_relevant": rel+=1
    return inc - exc + 0.3*rel

def aggregate_sign(match_json):
    inc=0; exc=0
    for c in match_json.get("criteria",[]):
        lab = (c.get("label") or "").lower()
        if lab=="include": inc+=1
        elif lab=="exclude": exc+=1
    s_elig = 1 if inc>exc else (-1 if exc>inc else 0)
    s_rel = 1 if inc>0 else 0
    return s_elig + 0.5*s_rel

def ndcg_at_k(pred_ids, rels, k=10):
    def dcg(items):
        s=0.0
        for i, docid in enumerate(items[:k], start=1):
            gain = rels.get(docid, 0)
            if gain>0:
                s += (gain)/math.log2(i+1)
        return s
    ideal = sorted(rels.items(), key=lambda x: -x[1])
    ideal_ids = [docid for docid,_ in ideal][:k]
    idcg = dcg(ideal_ids) or 1.0
    return dcg(pred_ids)/idcg

def precision_at_k(pred_ids, rels, k=10):
    hits = sum(1 for docid in pred_ids[:k] if rels.get(docid,0)>0)
    return hits/k

def main():
    ds = "sigir"
    model = "qwen-plus"
    token = register_and_login()
    corpus, id2doc = load_corpus(ds)
    queries = load_queries(ds, limit=5)
    qrels = load_qrels(ds)
    bm25 = SimpleBM25(corpus)
    rows=[]
    for q in queries:
        qid = q["_id"]; qtext = q["text"]
        cand_ids = bm25.search(qtext, top_k=50)
        scored_lin=[]; scored_sig=[]
        for cid in cand_ids[:10]:
            doc = id2doc[cid]
            inc_lines, exc_lines = extract_criteria(doc.get("text",""))
            crits=[]
            for l in inc_lines[:6]:
                if len(l)>3:
                    crits.append({"type":"inclusion","text":l})
            for l in exc_lines[:6]:
                if len(l)>3:
                    crits.append({"type":"exclusion","text":l})
            match = call_qwen_criterion(token, model, qtext, cid, crits)
            if not match:
                continue
            scored_lin.append((cid, aggregate_linear(match)))
            scored_sig.append((cid, aggregate_sign(match)))
        scored_lin.sort(key=lambda x: -x[1])
        scored_sig.sort(key=lambda x: -x[1])
        pred_lin = [cid for cid,_ in scored_lin[:10]]
        pred_sig = [cid for cid,_ in scored_sig[:10]]
        rels = qrels.get(qid, {})
        p10_lin = precision_at_k(pred_lin, rels, 10)
        ndcg_lin = ndcg_at_k(pred_lin, rels, 10)
        p10_sig = precision_at_k(pred_sig, rels, 10)
        ndcg_sig = ndcg_at_k(pred_sig, rels, 10)
        rows.append({
            "qid": qid,
            "linear_p@10": round(p10_lin,3),
            "linear_ndcg@10": round(ndcg_lin,3),
            "sign_p@10": round(p10_sig,3),
            "sign_ndcg@10": round(ndcg_sig,3)
        })
    avg = lambda arr, key: round(sum(x[key] for x in arr)/len(arr),3) if arr else 0.0
    summary = {
        "dataset": ds,
        "patients": len(rows),
        "linear_avg_p@10": avg(rows,"linear_p@10"),
        "linear_avg_ndcg@10": avg(rows,"linear_ndcg@10"),
        "sign_avg_p@10": avg(rows,"sign_p@10"),
        "sign_avg_ndcg@10": avg(rows,"sign_ndcg@10")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out = os.path.join(ROOT_DIR, "results_baseline_small.csv")
    if rows:
        import csv
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV exported: {out}")

if __name__=="__main__":
    main()
