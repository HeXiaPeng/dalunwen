import os
import re
import csv
import json
import math
from collections import Counter, defaultdict
from term_normalizer import load_synonyms, normalize_text, truncate_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset", "sigir")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_qrels(path):
    rel = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            parts = line.strip().split()
            if first and ("query-id" in (parts[0] if parts else "").lower()):
                first = False
                continue
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, docid, r = parts[0], parts[1], parts[2]
                try:
                    r = int(r)
                except:
                    r = 1
                if r > 0:
                    rel[str(qid)].add(str(docid))
    return rel

class WeightedBM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq = defaultdict(int)
        self.docs = []
        self.avgdl = 0.0

    @staticmethod
    def tokenize(text: str):
        text = (text or "").lower()
        tokens = []
        word = []
        for ch in text:
            if ch.isalnum():
                word.append(ch)
            else:
                if word:
                    tokens.append("".join(word))
                    word = []
        if word:
            tokens.append("".join(word))
        return tokens

    def add_document(self, doc_id: str, title: str, text: str):
        t_tokens = self.tokenize(title)
        x_tokens = self.tokenize(text)
        # section weighting: title x3; inclusion/exclusion lines x2
        inc_idx = (text or "").lower().find("inclusion criteria")
        exc_idx = (text or "").lower().find("exclusion criteria")
        inc_block = (text or "")[inc_idx:(exc_idx if exc_idx!=-1 else len(text))] if inc_idx!=-1 else ""
        exc_block = (text or "")[exc_idx:] if exc_idx!=-1 else ""
        inc_tokens = self.tokenize(inc_block)
        exc_tokens = self.tokenize(exc_block)
        tf = Counter()
        for tok in t_tokens:
            tf[tok] += 3
        for tok in x_tokens:
            tf[tok] += 1
        for tok in inc_tokens:
            tf[tok] += 2
        for tok in exc_tokens:
            tf[tok] += 2
        self.docs.append({"id": doc_id, "tf": tf, "len": sum(tf.values())})
        for term in tf.keys():
            self.doc_freq[term] += 1

    def consolidate(self):
        self.avgdl = (sum(d["len"] for d in self.docs) / len(self.docs)) if self.docs else 0.0

    def score(self, query_text: str, top_k: int = 10, query_boost=None):
        q_tokens = self.tokenize(query_text)
        # expand synonyms (minimal)
        syn_map = {
            "angina": ["chest", "pain"],
            "pneumonia": ["respiratory", "infection"],
            "mi": ["myocardial", "infarction"],
            "cad": ["coronary", "artery", "disease"],
        }
        expanded = []
        for t in q_tokens:
            expanded.append(t)
            for k, vs in syn_map.items():
                if t == k:
                    expanded.extend(vs)
        q_tf = Counter(expanded)
        # optional boost (e.g., extracted key terms)
        if query_boost:
            for t, w in query_boost.items():
                q_tf[t] += w
        N = len(self.docs)
        scores = []
        for d in self.docs:
            score = 0.0
            dl = d["len"] or 1
            for term, qcount in q_tf.items():
                df = self.doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                f = d["tf"].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                term_score = idf * (f * (self.k1 + 1)) / denom
                score += term_score
            scores.append((d["id"], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

AGE_RE = re.compile(r"(\d+)\s*[-]?\s*year[- ]?old", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(male|female)\b", re.IGNORECASE)

def ie_rule_score(query_text: str, doc_text: str):
    age_match = AGE_RE.search(query_text or "")
    age = int(age_match.group(1)) if age_match else None
    gender_match = GENDER_RE.search(query_text or "")
    gender = gender_match.group(1).lower() if gender_match else None
    inc_idx = (doc_text or "").lower().find("inclusion criteria")
    exc_idx = (doc_text or "").lower().find("exclusion criteria")
    inc_block = (doc_text or "")[inc_idx:(exc_idx if exc_idx!=-1 else len(doc_text))] if inc_idx!=-1 else ""
    exc_block = (doc_text or "")[exc_idx:] if exc_idx!=-1 else ""
    def check_age(c):
        c=c.lower()
        if age is None: return None
        m = re.findall(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*years", c)
        if m:
            lo, hi = int(m[0][0]), int(m[0][1])
            return lo <= age <= hi
        m1 = re.findall(r"age\s*(?:>=|≥|>)\s*(\d+)", c)
        m2 = re.findall(r"age\s*(?:<=|≤|<)\s*(\d+)", c)
        ok=None
        if m1: ok = age >= int(m1[0])
        if m2: ok = (ok if ok is not None else True) and (age <= int(m2[0]))
        return ok
    def check_gender(c):
        c=c.lower()
        if gender is None: return None
        if "both" in c or "all" in c: return True
        if "male" in c and gender=="male": return True
        if "female" in c and gender=="female": return True
        return None
    M_total=0; M_hit=0; R_vals=[]
    for line in inc_block.split("\n"):
        if any(k in line.lower() for k in ["age","years","male","female","both","all"]):
            M_total+=1
            aok = check_age(line)
            gok = check_gender(line)
            if aok is True or gok is True:
                M_hit+=1
            if aok is not None:
                R_vals.append(1.0 if aok else 0.0)
    excl_total=0; excl_safe=0
    for line in exc_block.split("\n"):
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
    return CCS

def precision_at_k(topk_ids, gold_set, k):
    hits = sum(1 for i in topk_ids if i in gold_set)
    return hits / k

def hitrate_at_k(topk_ids, gold_set):
    return 1 if any(i in gold_set for i in topk_ids) else 0

def main():
    corpus_path = os.path.join(DATA_DIR, "corpus.jsonl")
    queries_path = os.path.join(DATA_DIR, "queries.jsonl")
    qrels_path = os.path.join(DATA_DIR, "qrels", "test.tsv")
    # build index
    bm25 = WeightedBM25()
    syn_path = os.path.join(os.path.dirname(__file__), "resources", "synonyms.json")
    syn2canon = load_synonyms(syn_path)
    id_to_text = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            cid = entry["_id"]
            title = normalize_text(entry.get("title",""), syn2canon)
            text_raw = entry.get("text","")
            text_norm = normalize_text(text_raw, syn2canon)
            text = truncate_text(text_norm, max_chars=450)
            id_to_text[cid] = text
            bm25.add_document(cid, title, text)
    bm25.consolidate()
    # load
    queries = list(load_jsonl(queries_path))
    qrels = load_qrels(qrels_path)
    # evaluate
    k=10
    total_prec=0.0
    total_hit=0
    total_ccs=0.0
    rows=[]
    for q in queries:
        qid = q.get("_id")
        qtext = normalize_text(q.get("text") or "", syn2canon)
        # boosted query terms: extract simple disease words (heuristic)
        boost = {}
        for t in bm25.tokenize(qtext):
            if len(t) >= 6:
                boost[t] = boost.get(t, 0) + 1
        topk = bm25.score(qtext, top_k=50, query_boost=boost)
        # fuse with I/E CCS: re-rank top50 by bm25*(0.5+0.5*ccs)
        reranked = []
        for docid, s in topk:
            ccs = ie_rule_score(qtext, id_to_text.get(docid,""))
            reranked.append((docid, s*(0.5+0.5*ccs)))
        reranked.sort(key=lambda x:x[1], reverse=True)
        final_top = [docid for docid,_ in reranked[:k]]
        gold = qrels.get(qid, set())
        p_at_k = precision_at_k(final_top, gold, k)
        hit = hitrate_at_k(final_top, gold)
        total_prec += p_at_k
        total_hit += hit
        # record CCS (Top-1)
        top1_ccs = ie_rule_score(qtext, id_to_text.get(final_top[0],"")) if final_top else 0.0
        total_ccs += top1_ccs
        rows.append({
            "query_id": qid,
            "p_at_k": round(p_at_k,3),
            "hit_at_k": hit,
            "gold_count": len(gold),
            "top1_ccs": round(top1_ccs,3)
        })
    n = len(queries) if queries else 1
    print(json.dumps({
        "dataset":"sigir",
        "k":k,
        "queries":n,
        "avg_p@10": round(total_prec/n,3),
        "hit_rate@10": round(total_hit/n,3),
        "top1_avg_ccs": round(total_ccs/n,3)
    }, ensure_ascii=False, indent=2))
    out_path = os.path.join(os.path.dirname(__file__), "results_sigir_eval_opt.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV exported: {out_path}")

if __name__ == "__main__":
    main()
