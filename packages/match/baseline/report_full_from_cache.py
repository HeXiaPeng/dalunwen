import os
import json
import csv
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lib_dataset import load_dataset
from lib_aggregate import linear_aggregate, llm_aggregate, fusion_score
from lib_metrics import ndcg_at_k, p_at_k, roc_auc

BASE = os.path.dirname(__file__)
OUTPUT_ROOT = os.path.join(BASE, "output")
OUT_CSV = os.path.join(BASE, "trec2021_full_results.csv")
OUT_MD = os.path.join(BASE, "trec2021_full_report.md")

def iter_cached_qids():
    if not os.path.exists(OUTPUT_ROOT):
        return []
    qids = []
    for name in os.listdir(OUTPUT_ROOT):
        p = os.path.join(OUTPUT_ROOT, name, "index.json")
        if os.path.isfile(p):
            qids.append(name)
    return sorted(qids)

def load_index(qid):
    path = os.path.join(OUTPUT_ROOT, qid, "index.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def aggregate_for_qid(qid, index, gold):
    items = index.get("results", [])
    scored = {"linear": [], "llm": [], "fusion": []}
    y_true=[]; y_score_lin=[]; y_score_llm=[]; y_score_fus=[]
    for item in items:
        cid = item.get("trial_id")
        rs_lin, xs_lin, stats_lin = linear_aggregate(item)
        rs_llm, xs_llm, stats_llm = llm_aggregate(item)
        rs_fus = fusion_score(stats_lin, stats_llm)
        scored["linear"].append((cid, rs_lin))
        scored["llm"].append((cid, rs_llm))
        scored["fusion"].append((cid, rs_fus))
        gt = gold.get(qid, {}).get(cid, 0)
        y_true.append(1 if gt>=1 else 0)
        y_score_lin.append(xs_lin)
        y_score_llm.append(xs_llm)
        y_score_fus.append(rs_fus)
    for k in scored:
        scored[k].sort(key=lambda x: -x[1])
    pred_lin = [cid for cid,_ in scored["linear"]]
    pred_llm = [cid for cid,_ in scored["llm"]]
    pred_fus = [cid for cid,_ in scored["fusion"]]
    rels = gold.get(qid, {})
    rows = [
        {"qid": qid, "strategy":"linear", "p@10": round(p_at_k(pred_lin, rels, 10),3), "ndcg@10": round(ndcg_at_k(pred_lin, rels, 10),3), "auroc": round(roc_auc(y_true, y_score_lin),3)},
        {"qid": qid, "strategy":"llm",    "p@10": round(p_at_k(pred_llm, rels, 10),3), "ndcg@10": round(ndcg_at_k(pred_llm, rels, 10),3), "auroc": round(roc_auc(y_true, y_score_llm),3)},
        {"qid": qid, "strategy":"fusion", "p@10": round(p_at_k(pred_fus, rels, 10),3), "ndcg@10": round(ndcg_at_k(pred_fus, rels, 10),3), "auroc": round(roc_auc(y_true, y_score_fus),3)},
    ]
    return rows

def write_csv(rows):
    if not rows:
        return
    header = ["qid","strategy","p@10","ndcg@10","auroc"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def summarize(rows):
    agg={}
    for r in rows:
        s=r["strategy"]
        agg.setdefault(s, {"p":[], "n":[], "a":[]})
        agg[s]["p"].append(float(r["p@10"]))
        agg[s]["n"].append(float(r["ndcg@10"]))
        agg[s]["a"].append(float(r["auroc"]))
    out={}
    for s,v in agg.items():
        p=sum(v["p"])/len(v["p"]) if v["p"] else 0.0
        n=sum(v["n"])/len(v["n"]) if v["n"] else 0.0
        a=sum(v["a"])/len(v["a"]) if v["a"] else 0.0
        out[s]={"avg_p@10": round(p,3), "avg_ndcg@10": round(n,3), "avg_auroc": round(a,3), "overall_avg": round((p+n+a)/3,3)}
    return out

def write_md(summary, rows, total_patients):
    lines=[]
    lines.append("# TREC 2021 全量评测（缓存）")
    lines.append("")
    lines.append(f"总患者数（缓存数目）: {total_patients}")
    lines.append("")
    lines.append("| Strategy | Avg NDCG@10 | Avg P@10 | Avg AUROC | Overall |")
    lines.append("| --- | --- | --- | --- | --- |")
    for s,vals in summary.items():
        lines.append(f"| {s} | {vals['avg_ndcg@10']} | {vals['avg_p@10']} | {vals['avg_auroc']} | {vals['overall_avg']} |")
    lines.append("")
    lines.append("## 逐患者明细")
    lines.append("| qid | strategy | p@10 | ndcg@10 | auroc |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(f"| {r['qid']} | {r['strategy']} | {r['p@10']} | {r['ndcg@10']} | {r['auroc']} |")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    _, _, _, gold = load_dataset("trec_2021")
    qids = iter_cached_qids()
    all_rows=[]
    for qid in qids:
        index = load_index(qid)
        rows = aggregate_for_qid(qid, index, gold)
        all_rows.extend(rows)
    write_csv(all_rows)
    summary = summarize(all_rows)
    write_md(summary, all_rows, len(qids))
    print(json.dumps({"patients_cached": len(qids), "summary": summary, "csv": OUT_CSV, "md": OUT_MD}, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
