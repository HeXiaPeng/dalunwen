import os
import json
import math
import csv
import requests
from collections import Counter, defaultdict
import re
import time

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
            tf = Counter(self.tokenize(" ".join([doc.get("title",""), doc.get("text","")]).lower()))
            self.docs.append({"tf": tf, "len": sum(tf.values()), "id": doc.get("_id")})
            for t in tf.keys():
                self.df[t] += 1
        self.avgdl = sum(d["len"] for d in self.docs)/len(self.docs) if self.docs else 0.0
    @staticmethod
    def tokenize(text):
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
        qtf = Counter(self.tokenize((query or "").lower()))
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
    r = sess.post(f"{BASE_BACKEND}/api/users/register", json={"username":"sigir_full_cmp","password":"sigir_full_cmp_pwd"})
    if r.status_code == 200 and r.json().get("code")==200:
        token = r.json()["data"]["token"]
    else:
        r = sess.post(f"{BASE_BACKEND}/api/users/login", json={"username":"sigir_full_cmp","password":"sigir_full_cmp_pwd"})
        r.raise_for_status()
        token = r.json()["data"]["token"]
    return token

AGE_RE = re.compile(r"(\\d+)\\s*[-]?\\s*year[- ]?old", re.IGNORECASE)
GENDER_RE = re.compile(r"\\b(male|female)\\b", re.IGNORECASE)

def ie_rule_score(patient_text, doc_text):
    age_match = AGE_RE.search(patient_text or "")
    age = int(age_match.group(1)) if age_match else None
    gender_match = GENDER_RE.search(patient_text or "")
    gender = gender_match.group(1).lower() if gender_match else None
    inc_idx = (doc_text or "").lower().find("inclusion criteria")
    exc_idx = (doc_text or "").lower().find("exclusion criteria")
    inc_block = (doc_text or "")[inc_idx:(exc_idx if exc_idx!=-1 else len(doc_text))] if inc_idx!=-1 else ""
    exc_block = (doc_text or "")[exc_idx:] if exc_idx!=-1 else ""
    def check_age(c):
        c=c.lower()
        if age is None: return None
        m = re.findall(r"(\\d+)\\s*(?:-|to|–)\\s*(\\d+)\\s*years", c)
        if m:
            lo,hi = int(m[0][0]), int(m[0][1])
            return (lo<=age<=hi)
        m1 = re.findall(r"age\\s*(?:>=|≥|>)\\s*(\\d+)", c)
        m2 = re.findall(r"age\\s*(?:<=|≤|<)\\s*(\\d+)", c)
        ok=None
        if m1: ok = age>=int(m1[0])
        if m2: ok = (ok if ok is not None else True) and age<=int(m2[0])
        return ok
    def check_gender(c):
        c=c.lower()
        if gender is None: return None
        if "both" in c or "all" in c: return True
        if "male" in c and gender=="male": return True
        if "female" in c and gender=="female": return True
        return None
    M_total=0; M_hit=0; R_vals=[]
    for line in inc_block.split("\\n"):
        if any(k in line.lower() for k in ["age","years","male","female","both","all"]):
            M_total+=1
            aok = check_age(line)
            gok = check_gender(line)
            if aok is True or gok is True:
                M_hit+=1
            if aok is not None:
                R_vals.append(1.0 if aok else 0.0)
    excl_total=0; excl_safe=0
    for line in exc_block.split("\\n"):
        c=line.lower()
        violated=False
        aok = check_age(c)
        if aok is False: violated=True
        if "male" in c and gender=="female": violated=True
        if "female" in c and gender=="male": violated=True
        if any(k in c for k in ["age","years","male","female"]):
            excl_total+=1
        excl_safe += (0 if violated else 1)
    M = (M_hit/M_total) if M_total>0 else 0.0
    E = (excl_safe/excl_total) if excl_total>0 else 1.0
    R = sum(R_vals)/len(R_vals) if R_vals else 0.0
    CCS = 0.5*M + 0.3*E + 0.2*R
    return round(CCS,3)

def build_prompt(patient_text, candidates):
    lines=[]
    for doc in candidates:
        cid = doc["_id"]
        title = doc.get("title","")
        text = (doc.get("text","") or "").replace("\\n"," ").strip()[:300]
        lines.append(f"ID: {cid}, Title: {title}, Text: {text}")
    prompt = f"""
You are a clinical trial matching assistant. Given a patient case and a list of candidate clinical trials (ID, Title, Text), select the top 10 most suitable trials and assign each a score (0-100). Consider eligibility and relevance broadly.

Patient:
{(patient_text or '')[:800]}

Candidates:
{os.linesep.join(lines)}

Return STRICT JSON only:
{{"trials":[{{"trial_id":"...", "score":90, "reason":"..."}}]}}
"""
    return prompt

def main():
    start = time.time()
    token = register_and_login()
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    corpus, id2doc = load_corpus()
    bm25 = SimpleBM25(corpus)
    queries = load_queries()
    qrels = load_qrels()
    k=10
    rows=[]
    base_prec_sum=0.0; base_hit_sum=0
    llm_prec_sum=0.0; llm_hit_sum=0
    base_ccs_sum=0.0; llm_ccs_sum=0.0
    total = len(queries)
    print(f"Start SIGIR full compare: total_queries={total}, k={k}", flush=True)
    processed = 0
    for q in queries:
        qid = q.get("_id"); text = q.get("text") or ""
        gold = qrels.get(qid, set())
        # baseline top10 via BM25 over full corpus
        base_top_ids = [docid for docid,_ in bm25.search(text, top_k=k)]
        base_hits = sum(1 for tid in base_top_ids if tid in gold)
        base_prec = base_hits/k
        base_hit = 1 if base_hits>0 else 0
        base_prec_sum += base_prec; base_hit_sum += base_hit
        base_top1_ccs = 0.0
        if base_top_ids:
            base_top1_ccs = ie_rule_score(text, id2doc.get(base_top_ids[0], {}).get("text",""))
        base_ccs_sum += base_top1_ccs
        # paper method: LLM fine ranking over BM25 top50 candidates
        cand_ids = [docid for docid,_ in bm25.search(text, top_k=50)]
        candidates = [id2doc[i] for i in cand_ids if i in id2doc]
        prompt = build_prompt(text, candidates)
        payload = {"messages":[{"role":"system","content":"You output only valid JSON."},{"role":"user","content":prompt}]}
        r = requests.post(f"{BASE_BACKEND}/api/ai/generate", headers=headers, json=payload)
        llm_top_ids=[]
        if r.status_code==200:
            content = str(r.json().get("data","")).replace("```json","").replace("```","").strip()
            try:
                parsed = json.loads(content)
                trials = parsed["trials"] if isinstance(parsed, dict) else parsed
                llm_top_ids = [t.get("trial_id") for t in trials[:k] if t.get("trial_id")]
            except:
                llm_top_ids=[]
        llm_hits = sum(1 for tid in llm_top_ids if tid in gold)
        llm_prec = llm_hits/k
        llm_hit = 1 if llm_hits>0 else 0
        llm_prec_sum += llm_prec; llm_hit_sum += llm_hit
        llm_top1_ccs = 0.0
        if llm_top_ids:
            llm_top1_ccs = ie_rule_score(text, id2doc.get(llm_top_ids[0], {}).get("text",""))
        llm_ccs_sum += llm_top1_ccs
        rows.append({
            "query_id": qid,
            "gold_count": len(gold),
            "baseline_p@10": round(base_prec,3),
            "baseline_hit@10": base_hit,
            "baseline_top1_ccs": round(base_top1_ccs,3),
            "paper_p@10": round(llm_prec,3),
            "paper_hit@10": llm_hit,
            "paper_top1_ccs": round(llm_top1_ccs,3)
        })
        processed += 1
        if processed % 5 == 0 or processed == total:
            elapsed = time.time() - start
            avg_time = elapsed / processed if processed else 0
            eta = avg_time * (total - processed)
            print(
                f"[Progress] {processed}/{total} "
                f"elapsed={int(elapsed)}s eta={int(eta)}s "
                f"baseline_avg_p@10={round(base_prec_sum/processed,3)} "
                f"paper_avg_p@10={round(llm_prec_sum/processed,3)}",
                flush=True
            )
    n = len(queries) if queries else 1
    summary = {
        "dataset": "sigir",
        "k": k,
        "queries": n,
        "baseline_avg_p@10": round(base_prec_sum/n,3),
        "baseline_hit_rate@10": round(base_hit_sum/n,3),
        "baseline_top1_avg_ccs": round(base_ccs_sum/n,3),
        "paper_avg_p@10": round(llm_prec_sum/n,3),
        "paper_hit_rate@10": round(llm_hit_sum/n,3),
        "paper_top1_avg_ccs": round(llm_ccs_sum/n,3)
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    out_path = os.path.join(os.path.dirname(__file__), "results_sigir_compare_full.csv")
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV exported: {out_path}", flush=True)

if __name__=="__main__":
    main()
