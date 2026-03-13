import os
import re
import csv
import json
import math
import argparse
from collections import Counter, defaultdict

# ------------------------------
# Simple BM25 (no external deps)
# ------------------------------
class SimpleBM25:
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

    def add_document(self, doc_id: str, text: str):
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        self.docs.append({"id": doc_id, "tf": tf, "len": len(tokens)})
        for term in tf.keys():
            self.doc_freq[term] += 1

    def consolidate(self):
        self.avgdl = (sum(d["len"] for d in self.docs) / len(self.docs)) if self.docs else 0.0

    def score(self, query_text: str, top_k: int = 10):
        q_tokens = self.tokenize(query_text)
        q_tf = Counter(q_tokens)
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

# ------------------------------
# Loading utilities
# ------------------------------
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
            if first and ("query-id" in parts[0].lower()):
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

def build_doc_text(doc, dataset):
    if dataset == "sigir":
        return " ".join([doc.get("title", ""), doc.get("text", "")])
    # Fallback fields for trec datasets if available
    title = doc.get("brief_title") or doc.get("title") or ""
    summary = doc.get("brief_summary") or doc.get("text") or ""
    return " ".join([title, summary])

def get_doc_id(doc):
    return str(doc.get("_id") or doc.get("nct_id") or doc.get("id") or "")

def get_query_id(q):
    return str(q.get("_id") or q.get("qid") or q.get("id") or "")

def get_query_text(q):
    return q.get("text") or q.get("query") or ""

# ------------------------------
# Simple I/E rule scoring
# ------------------------------
AGE_RE = re.compile(r"(\d+)\s*[-]?\s*year[- ]?old", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(male|female)\b", re.IGNORECASE)

def extract_patient_profile(query_text: str):
    age_match = AGE_RE.search(query_text)
    age = int(age_match.group(1)) if age_match else None
    gender_match = GENDER_RE.search(query_text)
    gender = gender_match.group(1).lower() if gender_match else None
    return {"age": age, "gender": gender}

def parse_criteria(text: str):
    inc = []
    exc = []
    t = text or ""
    # naive split by markers
    inc_idx = t.lower().find("inclusion criteria")
    exc_idx = t.lower().find("exclusion criteria")
    if inc_idx != -1:
        inc_block = t[inc_idx: (exc_idx if exc_idx != -1 else len(t))]
        inc = [line.strip() for line in inc_block.split("\n") if len(line.strip()) >= 3]
    if exc_idx != -1:
        exc_block = t[exc_idx: len(t)]
        exc = [line.strip() for line in exc_block.split("\n") if len(line.strip()) >= 3]
    return inc, exc

def check_age_in_range(age, criterion):
    if age is None:
        return None
    m = re.findall(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*years", criterion.lower())
    if m:
        for a, b in m:
            lo, hi = int(a), int(b)
            if lo <= age <= hi:
                return True
            else:
                return False
    # single bound
    m2 = re.findall(r"age\s*(?:>=|≥|>)\s*(\d+)", criterion.lower())
    if m2 and age is not None:
        return age >= int(m2[0])
    m3 = re.findall(r"age\s*(?:<=|≤|<)\s*(\d+)", criterion.lower())
    if m3 and age is not None:
        return age <= int(m3[0])
    return None

def check_gender_match(gender, criterion):
    if gender is None:
        return None
    c = criterion.lower()
    if "male" in c and "female" in c:
        return True
    if "both" in c or "all" in c:
        return True
    if "male" in c and gender == "male":
        return True
    if "female" in c and gender == "female":
        return True
    # no explicit gender info
    return None

def ie_rule_score(query_text: str, doc_text: str):
    profile = extract_patient_profile(query_text)
    inc, exc = parse_criteria(doc_text)
    must_total = len(inc)
    excl_total = len(exc)
    must_hit = 0
    excl_safe = 0
    # range scoring
    ranges = []
    for c in inc:
        age_ok = check_age_in_range(profile["age"], c)
        gender_ok = check_gender_match(profile["gender"], c)
        if age_ok is True or gender_ok is True:
            must_hit += 1
        # collect ranges
        if age_ok is not None:
            ranges.append(1.0 if age_ok else 0.0)
    for c in exc:
        # if criterion mentions gender/age and we violate, count as triggered
        age_ok = check_age_in_range(profile["age"], c)
        gender_ok = check_gender_match(profile["gender"], c)
        violated = False
        if age_ok is False:
            violated = True
        if gender_ok is False:
            violated = True
        excl_safe += (0 if violated else 1)
    M = (must_hit / must_total) if must_total > 0 else 0.0
    E = (excl_safe / excl_total) if excl_total > 0 else 1.0
    R = sum(ranges) / len(ranges) if ranges else 0.0
    CCS = 0.5 * M + 0.3 * E + 0.2 * R
    # Eligibility pass
    epr_pass = (M >= 0.6 and E >= 0.9)
    evr_trigger = (E < 1.0)
    return {
        "M": M, "E": E, "R": R, "CCS": CCS,
        "EPR_pass": 1 if epr_pass else 0,
        "EVR_trigger": 1 if evr_trigger else 0
    }

# ------------------------------
# Metrics
# ------------------------------
def precision_at_k(topk_ids, gold_set, k):
    hits = sum(1 for i in topk_ids if i in gold_set)
    return hits / k

def hitrate_at_k(topk_ids, gold_set):
    return 1 if any(i in gold_set for i in topk_ids) else 0

# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="TrialGPT configurable evaluation")
    parser.add_argument("--dataset", choices=["sigir", "trec_2021", "trec_2022"], default="sigir")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--full_corpus", action="store_true", help="Use full corpus without truncation")
    parser.add_argument("--export_csv", type=str, default="results_trialgpt_eval.csv")
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset", args.dataset)
    corpus_path = os.path.join(base_dir, "corpus.jsonl")
    queries_path = os.path.join(base_dir, "queries.jsonl")
    qrels_path = os.path.join(base_dir, "qrels", "test.tsv")

    corpus = []
    for i, doc in enumerate(load_jsonl(corpus_path)):
        corpus.append(doc)
        if args.dataset == "sigir" and not args.full_corpus and i >= 3000:
            break

    bm25 = SimpleBM25(k1=1.2, b=0.75)
    id_to_text = {}
    for doc in corpus:
        doc_id = get_doc_id(doc)
        text = build_doc_text(doc, args.dataset)
        id_to_text[doc_id] = text
        bm25.add_document(doc_id, text)
    bm25.consolidate()

    queries = list(load_jsonl(queries_path))
    qrels = load_qrels(qrels_path)

    eval_queries = queries[:args.samples]

    total_prec = 0.0
    total_hit = 0
    total_epr = 0
    total_evr = 0
    total_ccs = 0.0
    rows = []

    print(f"=== TrialGPT evaluation (dataset={args.dataset}, k={args.k}, samples={args.samples}) ===")
    for q in eval_queries:
        qid = get_query_id(q)
        qtext = get_query_text(q)
        topk = bm25.score(qtext, top_k=args.k)
        top_ids = [docid for docid, _ in topk]
        gold = qrels.get(qid, set())
        p_at_k = precision_at_k(top_ids, gold, args.k)
        hit = hitrate_at_k(top_ids, gold)
        # I/E rule scoring on Top-1 doc (quick proxy)
        ie = {"M": 0.0, "E": 1.0, "R": 0.0, "CCS": 0.0, "EPR_pass": 0, "EVR_trigger": 0}
        if top_ids:
            doc_text = id_to_text.get(top_ids[0], "")
            ie = ie_rule_score(qtext, doc_text)
        total_prec += p_at_k
        total_hit += hit
        total_epr += ie["EPR_pass"]
        total_evr += ie["EVR_trigger"]
        total_ccs += ie["CCS"]

        print(f"\n[Query {qid}] {qtext[:120]}...")
        print(f"P@{args.k}: {p_at_k:.3f}, Hit@{args.k}: {hit}, gold={len(gold)}")
        print(f"Top-1 I/E => M={ie['M']:.2f}, E={ie['E']:.2f}, R={ie['R']:.2f}, CCS={ie['CCS']:.2f}, EPR_pass={ie['EPR_pass']}, EVR_trigger={ie['EVR_trigger']}")

        rows.append({
            "query_id": qid,
            "p_at_k": f"{p_at_k:.3f}",
            "hit_at_k": hit,
            "gold_count": len(gold),
            "top1_ccs": f"{ie['CCS']:.3f}",
            "top1_epr_pass": ie["EPR_pass"],
            "top1_evr_trigger": ie["EVR_trigger"]
        })

    avg_prec = total_prec / len(eval_queries) if eval_queries else 0.0
    hit_rate = total_hit / len(eval_queries) if eval_queries else 0.0
    avg_ccs = total_ccs / len(eval_queries) if eval_queries else 0.0
    epr_rate = total_epr / len(eval_queries) if eval_queries else 0.0
    evr_rate = total_evr / len(eval_queries) if eval_queries else 0.0

    print("\n=== Summary ===")
    print(f"Avg P@{args.k}: {avg_prec:.3f}")
    print(f"HitRate@{args.k}: {hit_rate:.3f}")
    print(f"Top-1 Avg CCS: {avg_ccs:.3f}")
    print(f"EPR_pass_rate (Top-1): {epr_rate:.3f}")
    print(f"EVR_trigger_rate (Top-1): {evr_rate:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), args.export_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV exported: {out_path}")

if __name__ == "__main__":
    main()
