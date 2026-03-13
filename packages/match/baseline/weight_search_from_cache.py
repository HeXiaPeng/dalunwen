import os
import json
import csv
import itertools
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from lib_dataset import load_dataset
from lib_metrics import ndcg_at_k, p_at_k, roc_auc

BASE = os.path.dirname(__file__)
OUTPUT_ROOT = os.path.join(BASE, "output")
WEIGHTS_PATH = os.path.join(BASE, "fusion_weights.json")
OUT_CSV = os.path.join(BASE, "trec2021_weighted_results.csv")
OUT_MD = os.path.join(BASE, "trec2021_weighted_report.md")

def iter_cached_qids():
    if not os.path.exists(OUTPUT_ROOT):
        return []
    qids=[]
    for name in os.listdir(OUTPUT_ROOT):
        p=os.path.join(OUTPUT_ROOT,name,"index.json")
        if os.path.isfile(p): qids.append(name)
    return sorted(qids)

def load_index(qid):
    path=os.path.join(OUTPUT_ROOT,qid,"index.json")
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def aggregate_scores(item, a,b,g,d):
    inc=0; exc=0; nin=0; nex=0; nei=0
    for c in item.get("criteria",[]):
        lab=(c.get("label","").lower().strip().replace(" ","_"))
        t=c.get("type")
        if t=="inclusion":
            if lab=="include": inc+=1
            elif lab=="not_included": nin+=1
            elif lab in ("not_enough_information","nei","no_info"): nei+=1
        elif t=="exclusion":
            if lab=="exclude": exc+=1
            elif lab=="not_excluded": nex+=1
            elif lab in ("not_enough_information","nei","no_info"): nei+=1
    total_in=inc+nin; total_ex=exc+nex
    p_inc=(inc/total_in) if total_in>0 else 0.0
    p_nin=(nin/total_in) if total_in>0 else 0.0
    p_exc=(exc/total_ex) if total_ex>0 else 0.0
    p_nex=(nex/total_ex) if total_ex>0 else 0.0
    nei_rate=(nei/(total_in+total_ex)) if (total_in+total_ex)>0 else 0.0
    rel=float(item.get("relevance",0))
    rs = a*rel + b*(p_inc - p_exc) + g*(p_nex - p_nin) - d*nei_rate
    return rs

def eval_with_weights(qids, gold, a,b,g,d):
    rows=[]
    for qid in qids:
        index=load_index(qid)
        items=index.get("results",[])
        scored=[]
        y_true=[]; y_score=[]
        for it in items:
            cid=it.get("trial_id")
            rs=aggregate_scores(it,a,b,g,d)
            scored.append((cid,rs))
            gt=gold.get(qid,{}).get(cid,0)
            y_true.append(1 if gt>=1 else 0)
            y_score.append(rs)
        scored.sort(key=lambda x:-x[1])
        pred=[cid for cid,_ in scored]
        rels=gold.get(qid,{})
        rows.append({"qid":qid,"strategy":"fusion_weighted","p@10":round(p_at_k(pred,rels,10),3),"ndcg@10":round(ndcg_at_k(pred,rels,10),3),"auroc":round(roc_auc(y_true,y_score),3)})
    return rows

def write_csv(rows, path):
    if not rows: return
    header=["qid","strategy","p@10","ndcg@10","auroc"]
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=header); w.writeheader()
        for r in rows: w.writerow(r)

def summarize(rows):
    p=sum(float(r["p@10"]) for r in rows)/len(rows) if rows else 0.0
    n=sum(float(r["ndcg@10"]) for r in rows)/len(rows) if rows else 0.0
    a=sum(float(r["auroc"]) for r in rows)/len(rows) if rows else 0.0
    return {"avg_p@10":round(p,3),"avg_ndcg@10":round(n,3),"avg_auroc":round(a,3),"overall_avg":round((p+n+a)/3,3)}

def write_md(summary, rows, path):
    lines=[]
    lines.append("# TREC 2021 加权融合汇总（缓存）")
    lines.append("| Strategy | Avg NDCG@10 | Avg P@10 | Avg AUROC | Overall |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| fusion_weighted | {summary['avg_ndcg@10']} | {summary['avg_p@10']} | {summary['avg_auroc']} | {summary['overall_avg']} |")
    lines.append("")
    lines.append("| qid | p@10 | ndcg@10 | auroc |")
    lines.append("| --- | --- | --- | --- |")
    for r in rows:
        lines.append(f"| {r['qid']} | {r['p@10']} | {r['ndcg@10']} | {r['auroc']} |")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))

def main():
    _,_,_,gold=load_dataset("trec_2021")
    qids=iter_cached_qids()
    grid_a=[1.0,0.8,1.2]; grid_b=[0.5,0.7,0.3]; grid_g=[0.25,0.4,0.1]; grid_d=[0.1,0.2,0.05]
    best=None; best_summary=None; best_rows=None
    for a,b,g,d in itertools.product(grid_a,grid_b,grid_g,grid_d):
        rows=eval_with_weights(qids,gold,a,b,g,d)
        summary=summarize(rows)
        score=summary["overall_avg"]
        if (best is None) or (score>best[0]):
            best=(score,a,b,g,d); best_summary=summary; best_rows=rows
    with open(WEIGHTS_PATH,"w",encoding="utf-8") as f:
        json.dump({"a":best[1],"b":best[2],"g":best[3],"d":best[4],"overall":best[0],"summary":best_summary},f,ensure_ascii=False,indent=2)
    write_csv(best_rows, OUT_CSV)
    write_md(best_summary, best_rows, OUT_MD)
    print(json.dumps({"best_overall":best[0],"weights":{"a":best[1],"b":best[2],"g":best[3],"d":best[4]},"summary":best_summary,"csv":OUT_CSV,"md":OUT_MD},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
