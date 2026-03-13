import os
import json
import csv
import math
from collections import defaultdict, Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset", "sigir")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_qrels():
    gold = defaultdict(dict)  # qid -> docid -> rel
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
    def search(self, query, top_k=10, k1=1.2, b=0.75):
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

def eval_bm25_small(sample_qids):
    corpus = list(load_jsonl(os.path.join(DATA_DIR, "corpus.jsonl")))
    id2doc = {d["_id"]: d for d in corpus}
    queries = list(load_jsonl(os.path.join(DATA_DIR, "queries.jsonl")))
    qmap = {q["_id"]: q["text"] for q in queries}
    gold = load_qrels()
    bm25 = SimpleBM25(corpus)
    metrics=[]
    for qid in sample_qids:
        pred = bm25.search(qmap[qid], top_k=10)
        rels = gold.get(qid, {})
        p10 = precision_at_k(pred, rels, k=10)
        ndcg = ndcg_at_k(pred, rels, k=10)
        metrics.append((p10, ndcg))
    avg_p10 = sum(m[0] for m in metrics)/len(metrics)
    avg_ndcg = sum(m[1] for m in metrics)/len(metrics)
    return round(avg_p10,3), round(avg_ndcg,3)

def read_llm_csv(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def eval_llm_from_csv(csv_path):
    gold = load_qrels()
    rows = read_llm_csv(csv_path)
    metrics=[]
    for row in rows:
        qid = row.get("qid") or row.get("query_id")
        pred_ids_str = row.get("pred_ids","[]")
        pred_ids = []
        s = pred_ids_str.strip()
        # try JSON first
        try:
            pred_ids = json.loads(s)
        except:
            # fallback: split and strip quotes
            raw = s.strip("[]")
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            for p in parts:
                p = p.strip().strip("'").strip('"')
                if p:
                    pred_ids.append(p)
        rels = gold.get(qid, {})
        p10 = precision_at_k(pred_ids, rels, k=10)
        ndcg = ndcg_at_k(pred_ids, rels, k=10)
        metrics.append((p10, ndcg))
    avg_p10 = sum(m[0] for m in metrics)/len(metrics) if metrics else 0.0
    avg_ndcg = sum(m[1] for m in metrics)/len(metrics) if metrics else 0.0
    return round(avg_p10,3), round(avg_ndcg,3)

def main():
    # sample qids used by small scripts (first 5)
    sample_qids = ["sigir-20141","sigir-20142","sigir-20143","sigir-20144","sigir-20145"]
    bm25_p10, bm25_ndcg = eval_bm25_small(sample_qids)
    max_p10, max_ndcg = eval_llm_from_csv(os.path.join(os.path.dirname(__file__), "results_sigir_llm_qwenmax.csv"))
    plus_p10, plus_ndcg = eval_llm_from_csv(os.path.join(os.path.dirname(__file__), "results_sigir_llm_qwenplus.csv"))
    table = [
        ["Method","NDCG@10","P@10"],
        ["BM25 baseline", bm25_ndcg, bm25_p10],
        ["LLM ranking (qwen-max)", max_ndcg, max_p10],
        ["LLM ranking (qwen-plus)", plus_ndcg, plus_p10],
    ]
    print(json.dumps({"table": table}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
