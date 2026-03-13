import os
import json
import math
import requests
from collections import defaultdict, Counter

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
    corpus = list(load_jsonl(os.path.join(DATA_DIR, "corpus.jsonl")))
    id2doc = {d["_id"]: d for d in corpus}
    return corpus, id2doc

def load_queries():
    return list(load_jsonl(os.path.join(DATA_DIR, "queries.jsonl")))

def load_qrels():
    gold = defaultdict(dict)
    path = os.path.join(DATA_DIR, "qrels", "test.tsv")
    with open(path, "r", encoding="utf-8") as f:
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
                gold[str(qid)][str(docid)] = rel
    return gold

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
    def search(self, query, top_k=20, k1=1.2, b=0.75):
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

def extract_criteria(text, max_inc=8, max_exc=8):
    t = (text or "")
    low = t.lower()
    inc_idx = low.find("inclusion criteria")
    exc_idx = low.find("exclusion criteria")
    inc_block = t[inc_idx:(exc_idx if exc_idx!=-1 else len(t))] if inc_idx!=-1 else ""
    exc_block = t[exc_idx:] if exc_idx!=-1 else ""
    inc_lines = [l.strip() for l in inc_block.split("\n") if l.strip()]
    exc_lines = [l.strip() for l in exc_block.split("\n") if l.strip()]
    return inc_lines[:max_inc], exc_lines[:max_exc]

def call_qwen_match(token, model, patient_text, trial_id, inc_lines, exc_lines):
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    crits=[]
    for l in inc_lines:
        if len(l)>3:
            crits.append({"type":"inclusion","text":l})
    for l in exc_lines:
        if len(l)>3:
            crits.append({"type":"exclusion","text":l})
    prompt = {
        "patient": patient_text[:800],
        "trial_id": trial_id,
        "criteria": crits
    }
    sys = "You are a clinical trial matching assistant. For each criterion, label 'include' or 'exclude' or 'not_relevant', and provide brief evidence from patient text. Return STRICT JSON: {\"trial_id\":\"...\",\"criteria\":[{\"type\":\"inclusion\",\"text\":\"...\",\"label\":\"include|exclude|not_relevant\",\"evidence\":[\"...\"]}...]}"
    payload = {"model": model, "messages":[{"role":"system","content":sys},{"role":"user","content":json.dumps(prompt)}]}
    r = requests.post(f"{BASE_BACKEND}/api/ai/generate", headers=headers, json=payload)
    if r.status_code!=200:
        return None
    content = str(r.json().get("data","")).replace("```json","").replace("```","").strip()
    try:
        parsed = json.loads(content)
        return parsed
    except:
        return None

def aggregate_scores_linear(match_json):
    inc=0; exc=0; rel=0
    for c in match_json.get("criteria",[]):
        lab = (c.get("label") or "").lower()
        if lab=="include": inc+=1; rel+=1
        elif lab=="exclude": exc+=1
        elif lab=="not_relevant": rel+=1
    elig_score = inc - exc
    return elig_score, rel

def aggregate_scores_sign(match_json):
    inc=0; exc=0
    for c in match_json.get("criteria",[]):
        lab = (c.get("label") or "").lower()
        if lab=="include": inc+=1
        elif lab=="exclude": exc+=1
    s_elig = 1 if inc>exc else (-1 if exc>inc else 0)
    s_rel = 1 if inc>0 else 0
    return s_elig, s_rel

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

def register_and_login():
    sess = requests.Session()
    r = sess.post(f"{BASE_BACKEND}/api/users/register", json={"username":"trialgpt_match","password":"trialgpt_match_pwd"})
    if r.status_code==200 and r.json().get("code")==200:
        token = r.json()["data"]["token"]
    else:
        r = sess.post(f"{BASE_BACKEND}/api/users/login", json={"username":"trialgpt_match","password":"trialgpt_match_pwd"})
        r.raise_for_status()
        token = r.json()["data"]["token"]
    return token

def main():
    token = register_and_login()
    model = "qwen-plus"
    corpus, id2doc = load_corpus()
    queries = load_queries()[:3]
    qrels = load_qrels()
    bm25 = SimpleBM25(corpus)
    k=10
    results=[]
    for q in queries:
        qid = q["_id"]; qtext = q["text"]
        cand_ids = bm25.search(qtext, top_k=20)
        scored_linear=[]
        scored_sign=[]
        for cid in cand_ids:
            doc = id2doc[cid]
            inc_lines, exc_lines = extract_criteria(doc.get("text",""))
            match = call_qwen_match(token, model, qtext, cid, inc_lines, exc_lines)
            if not match:
                continue
            elig_score, rel_score = aggregate_scores_linear(match)
            s_elig, s_rel = aggregate_scores_sign(match)
            scored_linear.append((cid, elig_score + 0.3*rel_score))
            scored_sign.append((cid, s_elig + 0.5*s_rel))
        scored_linear.sort(key=lambda x: -x[1])
        scored_sign.sort(key=lambda x: -x[1])
        pred_linear = [cid for cid,_ in scored_linear[:k]]
        pred_sign = [cid for cid,_ in scored_sign[:k]]
        rels = qrels.get(qid, {})
        res = {
            "qid": qid,
            "linear_p@10": round(precision_at_k(pred_linear, rels, k),3),
            "linear_ndcg@10": round(ndcg_at_k(pred_linear, rels, k),3),
            "sign_p@10": round(precision_at_k(pred_sign, rels, k),3),
            "sign_ndcg@10": round(ndcg_at_k(pred_sign, rels, k),3),
        }
        results.append(res)
    avg = lambda arr, key: round(sum(x[key] for x in arr)/len(arr),3) if arr else 0.0
    summary = {
        "queries": len(results),
        "linear_avg_p@10": avg(results,"linear_p@10"),
        "linear_avg_ndcg@10": avg(results,"linear_ndcg@10"),
        "sign_avg_p@10": avg(results,"sign_p@10"),
        "sign_avg_ndcg@10": avg(results,"sign_ndcg@10"),
        "details": results
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
