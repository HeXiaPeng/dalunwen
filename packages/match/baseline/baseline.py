import os
import json
import math
import csv
from collections import defaultdict, Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aliy_client import AliyClient
from term_normalizer import load_synonyms, normalize_text, truncate_text

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "TrialGPT", "dataset", "sigir")
STATE_PATH = os.path.join(os.path.dirname(__file__), "run_state.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results_small.csv")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_corpus():
    path = os.path.join(DATA_DIR, "corpus.jsonl")
    raw = list(load_jsonl(path))
    syn2canon = load_synonyms(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "synonyms.json"))
    corpus=[]
    id2doc={}
    for d in raw:
        cid = d["_id"]
        title = normalize_text(d.get("title",""), syn2canon)
        text = normalize_text(d.get("text",""), syn2canon)
        text = truncate_text(text, max_chars=450)
        nd = {"_id": cid, "title": title, "text": text}
        corpus.append(nd)
        id2doc[cid] = nd
    return corpus, id2doc

def load_queries(limit=5):
    path = os.path.join(DATA_DIR, "queries.jsonl")
    syn2canon = load_synonyms(os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "synonyms.json"))
    qs=[]
    for q in load_jsonl(path):
        qs.append({"_id": q["_id"], "text": normalize_text(q.get("text",""), syn2canon)})
    return qs[:limit]

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
            tf = Counter(self.tokenize((" ".join([doc.get("title",""), doc.get("text","")])).lower()))
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
    def search(self, query, top_k=50, k1=1.2, b=0.75):
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
        return scores[:top_k]

def extract_criteria(text, max_inc=12, max_exc=12):
    t = (text or "")
    low = t.lower()
    inc_idx = low.find("inclusion criteria")
    exc_idx = low.find("exclusion criteria")
    inc_block = t[inc_idx:(exc_idx if exc_idx!=-1 else len(t))] if inc_idx!=-1 else ""
    exc_block = t[exc_idx:] if exc_idx!=-1 else ""
    inc_lines = [l.strip() for l in inc_block.split("\n") if l.strip()]
    exc_lines = [l.strip() for l in exc_block.split("\n") if l.strip()]
    return inc_lines[:max_inc], exc_lines[:max_exc]

AGE_RE = None
GENDER_RE = None
import re
AGE_RE = re.compile(r"(\d+)\s*[-]?\s*year[- ]?old", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(male|female)\b", re.IGNORECASE)

def ie_rule_score(patient_text, doc_text):
    age_match = AGE_RE.search(patient_text or "")
    age = int(age_match.group(1)) if age_match else None
    gender_match = GENDER_RE.search(patient_text or "")
    gender = gender_match.group(1).lower() if gender_match else None
    low = (doc_text or "").lower()
    inc_idx = low.find("inclusion criteria")
    exc_idx = low.find("exclusion criteria")
    inc_block = (doc_text or "")[inc_idx:(exc_idx if exc_idx!=-1 else len(doc_text))] if inc_idx!=-1 else ""
    exc_block = (doc_text or "")[exc_idx:] if exc_idx!=-1 else ""
    M_total=0; M_hit=0; R_vals=[]
    for line in inc_block.split("\n"):
        c=line.lower()
        if any(k in c for k in ["age","years","male","female","both","all"]):
            M_total+=1
            aok=None
            if age is not None:
                m = re.findall(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*years", c)
                if m:
                    lo,hi = int(m[0][0]), int(m[0][1]); aok = (lo<=age<=hi)
                m1 = re.findall(r"age\s*(?:>=|≥|>)\s*(\d+)", c)
                m2 = re.findall(r"age\s*(?:<=|≤|<)\s*(\d+)", c)
                if m1: aok = age>=int(m1[0])
                if m2: aok = (aok if aok is not None else True) and age<=int(m2[0])
            gok=None
            if gender is not None:
                if "both" in c or "all" in c: gok=True
                elif ("male" in c and gender=="male") or ("female" in c and gender=="female"): gok=True
            if aok is True or gok is True:
                M_hit+=1
            if aok is not None:
                R_vals.append(1.0 if aok else 0.0)
    excl_total=0; excl_safe=0
    for line in exc_block.split("\n"):
        c=line.lower()
        violated=False
        if gender is not None:
            if "male" in c and gender=="female": violated=True
            if "female" in c and gender=="male": violated=True
        if age is not None and ("age" in c or "years" in c):
            excl_total+=1
            m = re.findall(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*years", c)
            if m:
                lo,hi = int(m[0][0]), int(m[0][1])
                if not (lo<=age<=hi): violated=True
            m1 = re.findall(r"age\s*(?:>=|≥|>)\s*(\d+)", c)
            if m1 and age<int(m1[0]): violated=True
            m2 = re.findall(r"age\s*(?:<=|≤|<)\s*(\d+)", c)
            if m2 and age>int(m2[0]): violated=True
        elif any(k in c for k in ["male","female"]):
            excl_total+=1
        excl_safe += (0 if violated else 1)
    M = (M_hit/M_total) if M_total>0 else 0.0
    E = (excl_safe/excl_total) if excl_total>0 else 1.0
    R = sum(R_vals)/len(R_vals) if R_vals else 0.0
    CCS = 0.5*M + 0.3*E + 0.2*R
    return CCS

def call_qwen_criterion(client, model, patient_text, trial_id, title, snippet, crits):
    sys = "You are a clinical trial matching assistant. For each criterion, return label include|exclude|not_relevant and evidence. Return STRICT JSON."
    payload = {"patient": patient_text[:1200], "trial_id": trial_id, "title": title, "summary": snippet[:300], "criteria": crits}
    messages = [{"role":"system","content":sys},{"role":"user","content":json.dumps(payload)}]
    content = client.chat_completions(messages=messages, model=model)
    content = str(content).replace("```json","").replace("```","").strip()
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

def load_state():
    if not os.path.exists(STATE_PATH):
        return {"processed_qids": []}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def append_results_row(row, header):
    exists = os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

def main():
    client = AliyClient()
    model = "qwen-plus"
    corpus, id2doc = load_corpus()
    queries = load_queries(limit=5)
    qrels = load_qrels()
    bm25 = SimpleBM25(corpus)
    state = load_state()
    header = ["qid","linear_p@10","linear_ndcg@10","sign_p@10","sign_ndcg@10"]
    total = len(queries)
    processed_count = 0
    for q in queries:
        qid = q["_id"]
        if qid in state.get("processed_qids", []):
            continue
        qtext = q["text"]
        cand_scores = bm25.search(qtext, top_k=50)
        reranked=[]
        for cid, s in cand_scores:
            doc = id2doc[cid]
            ccs = ie_rule_score(qtext, doc.get("text",""))
            reranked.append((cid, s*(0.5+0.5*ccs)))
        reranked.sort(key=lambda x: -x[1])
        scored_lin=[]; scored_sig=[]
        for cid, _ in reranked[:15]:
            doc = id2doc[cid]
            inc_lines, exc_lines = extract_criteria(doc.get("text",""))
            crits=[]
            for l in inc_lines[:6]:
                if len(l)>3:
                    crits.append({"type":"inclusion","text":l})
            for l in exc_lines[:6]:
                if len(l)>3:
                    crits.append({"type":"exclusion","text":l})
            snippet = doc.get("text","")
            match = call_qwen_criterion(client, model, qtext, cid, doc.get("title",""), snippet, crits)
            if not match:
                continue
            scored_lin.append((cid, aggregate_linear(match)))
            scored_sig.append((cid, aggregate_sign(match)))
        scored_lin.sort(key=lambda x: -x[1])
        scored_sig.sort(key=lambda x: -x[1])
        pred_lin = [cid for cid,_ in scored_lin[:10]]
        pred_sig = [cid for cid,_ in scored_sig[:10]]
        rels = qrels.get(qid, {})
        row = {
            "qid": qid,
            "linear_p@10": round(precision_at_k(pred_lin, rels, 10),3),
            "linear_ndcg@10": round(ndcg_at_k(pred_lin, rels, 10),3),
            "sign_p@10": round(precision_at_k(pred_sig, rels, 10),3),
            "sign_ndcg@10": round(ndcg_at_k(pred_sig, rels, 10),3),
        }
        append_results_row(row, header)
        state.setdefault("processed_qids", []).append(qid)
        save_state(state)
        processed_count += 1
        print(f"{processed_count}/{total}")
    print(f"Processed {processed_count}/{total}")

if __name__=="__main__":
    main()
