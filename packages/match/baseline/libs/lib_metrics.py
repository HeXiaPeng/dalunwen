import math

def ndcg_at_k(pred_ids, rels, k=10):
    def dcg(items):
        s=0.0
        for i, docid in enumerate(items[:k], start=1):
            gain = rels.get(docid, 0)
            s += ((2**gain -1))/math.log2(i+1)
        return s
    ideal = sorted(rels.items(), key=lambda x: -x[1])
    ideal_ids = [docid for docid,_ in ideal][:k]
    idcg = dcg(ideal_ids) or 1.0
    return dcg(pred_ids)/idcg

def p_at_k(pred_ids, rels, k=10, threshold=1):
    hits = sum(1 for d in pred_ids[:k] if rels.get(d,0)>=threshold)
    return hits/k

def roc_auc(y_true, y_score):
    # y_true should be binary 0/1
    pairs=0; better=0; ties=0
    pos=[y_score[i] for i in range(len(y_true)) if y_true[i]==1]
    neg=[y_score[i] for i in range(len(y_true)) if y_true[i]==0]
    for ps in pos:
        for ns in neg:
            pairs+=1
            if ps>ns: better+=1
            elif ps==ns: ties+=1
    return (better + 0.5*ties)/pairs if pairs>0 else 0.0

def bar(current, total, width=24):
    total = max(total, 1)
    filled = int(width * current / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {current}/{total}"
