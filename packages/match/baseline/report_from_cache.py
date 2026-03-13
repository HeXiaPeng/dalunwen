import os
import csv
import json

BASE = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE, "framework_results.csv")
MD_PATH = os.path.join(BASE, "final_report.md")

def load_rows():
    if not os.path.exists(CSV_PATH):
        return []
    rows=[]
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def aggregate(rows):
    agg={}
    for row in rows:
        s=row["strategy"]
        agg.setdefault(s, {"p":[], "n":[], "a":[]})
        try:
            agg[s]["p"].append(float(row["p@10"]))
            agg[s]["n"].append(float(row["ndcg@10"]))
            agg[s]["a"].append(float(row["auroc"]))
        except:
            pass
    out={}
    for s,v in agg.items():
        p = sum(v["p"])/len(v["p"]) if v["p"] else 0.0
        n = sum(v["n"])/len(v["n"]) if v["n"] else 0.0
        a = sum(v["a"])/len(v["a"]) if v["a"] else 0.0
        out[s] = {
            "avg_p@10": round(p,3),
            "avg_ndcg@10": round(n,3),
            "avg_auroc": round(a,3),
            "overall_avg": round((p+n+a)/3,3)
        }
    return out

def write_md(summary, rows):
    lines=[]
    lines.append("# TREC 2021 小样本（5位患者，Top500）评测汇总（缓存）")
    lines.append("")
    lines.append("| Strategy | Avg NDCG@10 | Avg P@10 | Avg AUROC | Overall |")
    lines.append("| --- | --- | --- | --- | --- |")
    for s,vals in summary.items():
        lines.append(f"| {s} | {vals['avg_ndcg@10']} | {vals['avg_p@10']} | {vals['avg_auroc']} | {vals['overall_avg']} |")
    lines.append("")
    lines.append("## 逐患者明细（缓存）")
    lines.append("| qid | strategy | p@10 | ndcg@10 | auroc |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(f"| {row['qid']} | {row['strategy']} | {row['p@10']} | {row['ndcg@10']} | {row['auroc']} |")
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    rows = load_rows()
    summary = aggregate(rows)
    write_md(summary, rows)
    print(json.dumps({"summary": summary, "rows_count": len(rows), "report_path": MD_PATH}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
