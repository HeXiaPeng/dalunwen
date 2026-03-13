import json
import os
import math
from collections import Counter, defaultdict

# Minimal BM25-style scorer without external dependencies
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
        # update df per term once per doc
        for term in tf.keys():
            self.doc_freq[term] += 1

    def consolidate(self):
        if not self.docs:
            self.avgdl = 0.0
        else:
            self.avgdl = sum(d["len"] for d in self.docs) / len(self.docs)

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


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_qrels(path):
    # TREC/TrialGPT qrels: query_id \t doc_id \t relevance
    rel = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                # maybe tsv: qid \t docid \t rel
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


def build_doc_text(doc):
    # SIGIR subset fields: _id, title, text
    title = doc.get("title", "")
    text = doc.get("text", "")
    return " ".join([title, text])


def main():
    base_sigir = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset", "sigir")
    corpus_path = os.path.join(base_sigir, "corpus.jsonl")
    queries_path = os.path.join(base_sigir, "queries.jsonl")
    qrels_path = os.path.join(base_sigir, "qrels", "test.tsv")

    # Load small subset of corpus (to save time)
    corpus = []
    for i, doc in enumerate(load_jsonl(corpus_path)):
        corpus.append(doc)
        if i >= 3000:  # take first ~3000 docs for quick run
            break

    # Build BM25 index
    bm25 = SimpleBM25(k1=1.2, b=0.75)
    for doc in corpus:
        doc_id = str(doc.get("_id") or doc.get("nct_id") or doc.get("id") or "")
        bm25.add_document(doc_id, build_doc_text(doc))
    bm25.consolidate()

    # Load queries and qrels
    queries = list(load_jsonl(queries_path))
    qrels = load_qrels(qrels_path)

    # Evaluate on a few queries (avoid full run)
    eval_queries = queries[:5]
    k = 10

    total_hits = 0
    total_prec = 0.0

    print("=== TrialGPT quick BM25 validation (subset) ===")
    for q in eval_queries:
        qid = str(q.get("_id") or q.get("qid") or q.get("id") or "")
        qtext = q.get("text") or q.get("query") or ""
        topk = bm25.score(qtext, top_k=k)
        gold = qrels.get(qid, set())
        hits = sum(1 for docid, _ in topk if docid in gold)
        prec = hits / k
        total_hits += 1 if hits > 0 else 0
        total_prec += prec
        print(f"\n[Query {qid}] {qtext[:120]}...")
        print(f"Hit@{k}: {1 if hits>0 else 0}, P@{k}: {prec:.2f}, gold count: {len(gold)}")
        for rank, (docid, score) in enumerate(topk, start=1):
            print(f"  #{rank:<2} {docid}  score={score:.3f}")

    avg_prec = total_prec / len(eval_queries) if eval_queries else 0.0
    hit_rate = total_hits / len(eval_queries) if eval_queries else 0.0
    print("\n=== Summary ===")
    print(f"Avg P@{k}: {avg_prec:.3f}")
    print(f"HitRate@{k}: {hit_rate:.3f}")


if __name__ == "__main__":
    main()
