import os
import json
import sys
import csv
sys.path.append(os.path.dirname(__file__))
from libs.lib_dataset import load_dataset
from libs.lib_aggregate import linear_aggregate, llm_aggregate

def run_error_analysis(ds="trec_2021", top_k=10, output_file="baseline/results/error_analysis.csv"):
    print(f"Loading dataset {ds}...")
    corpus, id2doc, queries, gold, q2candidates = load_dataset(ds)
    
    # Locate results
    base_dir = os.path.join(os.path.dirname(__file__), "results", "batches", ds)
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.path.dirname(__file__), "results", "batches")
        
    errors = []
    processed_count = 0
    
    print(f"Analyzing errors in {base_dir}...")
    
    # Only process queries we have results for
    target_qids = [q["_id"] for q in queries]
    
    for qid in target_qids:
        # Load results
        q_dir = os.path.join(base_dir, qid)
        if not os.path.exists(q_dir):
            continue
            
        combined_results = []
        try:
            for fname in os.listdir(q_dir):
                if fname.endswith("_result.json"):
                    with open(os.path.join(q_dir, fname), "r", encoding="utf-8") as f:
                        res = json.load(f)
                        if res and "results" in res:
                            combined_results.extend(res["results"])
        except: continue
        
        if not combined_results: continue
        processed_count += 1
        
        # Calculate scores and sort
        scored = []
        for item in combined_results:
            cid = item.get("trial_id")
            if not cid: continue
            
            _, _, stats_lin = linear_aggregate(item)
            _, _, stats_llm = llm_aggregate(item)
            
            # Recalculate stats like in ablation (Full Model)
            inc = stats_lin.get("inc", 0)
            nin = stats_lin.get("nin", 0)
            exc = stats_lin.get("exc", 0)
            nex = stats_lin.get("nex", 0)
            nei = stats_lin.get("nei", 0)
            
            def safe_div(n, d): return n/d if d>0 else 0.0
            
            total_in = inc + nin
            total_ex = exc + nex
            total_all = inc + nin + exc + nex + nei
            
            p_inc = safe_div(inc, total_in)
            p_nin = safe_div(nin, total_in)
            p_exc = safe_div(exc, total_ex)
            p_nex = safe_div(nex, total_ex)
            nei_rate = safe_div(nei, total_all)
            
            rel = stats_llm.get("rel", 0)/100.0
            
            # Full Model Formula
            score = rel + 0.5*(p_inc - p_exc) + 0.25*(p_nex - p_nin) - 0.1*nei_rate
            
            scored.append({
                "cid": cid,
                "score": score,
                "rel": stats_llm.get("rel"),
                "criteria": item.get("criteria", []),
                "stats_lin": stats_lin
            })
            
        scored.sort(key=lambda x: -x["score"])
        top_items = scored[:top_k]
        
        # Analyze False Positives in Top K
        for rank, item in enumerate(top_items):
            cid = item["cid"]
            gt = gold.get(qid, {}).get(cid, 0)
            
            if gt == 0: # False Positive
                # Extract reasons (Top 3 positive evidence)
                reasons = []
                for c in item["criteria"]:
                    lab = c.get("label", "").lower()
                    if lab in ["include", "not_excluded"]:
                        ev = c.get("evidence", "").replace("\n", " ").strip()
                        if len(ev) > 100: ev = ev[:100] + "..."
                        reasons.append(f"[{lab.upper()}] {ev}")
                
                errors.append({
                    "qid": qid,
                    "rank": rank + 1,
                    "trial_id": cid,
                    "model_score": f"{item['score']:.3f}",
                    "llm_rel": item["rel"],
                    "gt_label": gt,
                    "error_type": "False Positive",
                    "model_reasoning": " | ".join(reasons[:3])
                })
                
    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "rank", "trial_id", "model_score", "llm_rel", "gt_label", "error_type", "model_reasoning"])
        writer.writeheader()
        writer.writerows(errors)
        
    print(f"Processed {processed_count} patients.")
    print(f"Found {len(errors)} False Positives in Top {top_k}. Saved to {output_file}")

if __name__ == "__main__":
    run_error_analysis()
