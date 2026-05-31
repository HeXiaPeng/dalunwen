import os
import json
import sys
sys.path.append(os.path.dirname(__file__))
from libs.lib_dataset import load_dataset
from libs.lib_metrics import ndcg_at_k, p_at_k, roc_auc
from libs.lib_aggregate import linear_aggregate, llm_aggregate

# Define ablation strategies
def fusion_full(stats_lin, stats_llm):
    rel = stats_llm.get("rel", 0.0) / 100.0
    return rel + 0.5*(stats_lin.get("p_inc",0.0) - stats_lin.get("p_exc",0.0)) + 0.25*(stats_lin.get("p_nex",0.0) - stats_lin.get("p_nin",0.0)) - 0.1*stats_lin.get("nei_rate",0.0)

def fusion_no_strong(stats_lin, stats_llm):
    rel = stats_llm.get("rel", 0.0) / 100.0
    # Remove 0.5 * (p_inc - p_exc)
    return rel + 0.25*(stats_lin.get("p_nex",0.0) - stats_lin.get("p_nin",0.0)) - 0.1*stats_lin.get("nei_rate",0.0)

def fusion_no_weak(stats_lin, stats_llm):
    rel = stats_llm.get("rel", 0.0) / 100.0
    # Remove 0.25 * (p_nex - p_nin)
    return rel + 0.5*(stats_lin.get("p_inc",0.0) - stats_lin.get("p_exc",0.0)) - 0.1*stats_lin.get("nei_rate",0.0)

def fusion_no_penalty(stats_lin, stats_llm):
    rel = stats_llm.get("rel", 0.0) / 100.0
    # Remove -0.1 * nei_rate
    return rel + 0.5*(stats_lin.get("p_inc",0.0) - stats_lin.get("p_exc",0.0)) + 0.25*(stats_lin.get("p_nex",0.0) - stats_lin.get("p_nin",0.0))

STRATEGIES = {
    "Full Model": fusion_full,
    "w/o Strong": fusion_no_strong,
    "w/o Weak": fusion_no_weak,
    "w/o Penalty": fusion_no_penalty
}

def run_ablation(ds="trec_2021", samples=1000):
    print(f"Loading dataset {ds}...")
    corpus, id2doc, queries, gold, q2candidates = load_dataset(ds)
    
    # Storage for metrics
    metrics = {k: {"p": [], "ps": [], "n": [], "a": [], "as": []} for k in STRATEGIES}
    
    processed = 0
    base_paths = [
        os.path.join(os.path.dirname(__file__), "results", "batches", ds),
        os.path.join(os.path.dirname(__file__), "results", "batches")
    ]
    
    print(f"Scanning for results...")
    
    target_qids = [q["_id"] for q in queries[:samples]]
    
    for qid in target_qids:
        found_dir = None
        for base in base_paths:
            candidate = os.path.join(base, qid)
            if os.path.exists(candidate):
                found_dir = candidate
                break
        
        if not found_dir:
            continue
            
        combined_results = []
        try:
            for fname in os.listdir(found_dir):
                if fname.endswith("_result.json"):
                    with open(os.path.join(found_dir, fname), "r", encoding="utf-8") as f:
                        res = json.load(f)
                        if res and "results" in res:
                            combined_results.extend(res["results"])
        except Exception as e:
            print(f"Error reading {qid}: {e}")
            continue
        
        if not combined_results:
            continue
            
        processed += 1
        
        item_stats = []
        for item in combined_results:
            cid = item.get("trial_id")
            gt = gold.get(qid, {}).get(cid, 0)
            y_rel = 1 if gt >= 1 else 0
            y_str = 1 if gt >= 2 else 0
            
            _, _, stats_lin = linear_aggregate(item)
            _, _, stats_llm = llm_aggregate(item)
            
            # Calculate ratios needed for the paper's formula
            inc = stats_lin.get("inc", 0)
            nin = stats_lin.get("nin", 0)
            exc = stats_lin.get("exc", 0)
            nex = stats_lin.get("nex", 0)
            nei = stats_lin.get("nei", 0)
            
            def safe_div(n, d): return n/d if d>0 else 0.0
            
            total_in = inc + nin
            total_ex = exc + nex
            total_all = inc + nin + exc + nex + nei
            
            stats_lin["p_inc"] = safe_div(inc, total_in)
            stats_lin["p_nin"] = safe_div(nin, total_in)
            stats_lin["p_exc"] = safe_div(exc, total_ex)
            stats_lin["p_nex"] = safe_div(nex, total_ex)
            stats_lin["nei_rate"] = safe_div(nei, total_all)
            
            item_stats.append({
                "cid": cid,
                "stats_lin": stats_lin,
                "stats_llm": stats_llm,
                "y_rel": y_rel,
                "y_str": y_str
            })
            
        # DEBUG: Print first item of first patient
        if processed == 1 and item_stats:
            it = item_stats[0]
            print(f"DEBUG: cid={it['cid']}, rel={it['stats_llm'].get('rel')}, lin={it['stats_lin']}")
            for s_name, s_func in STRATEGIES.items():
                print(f"  {s_name}: {s_func(it['stats_lin'], it['stats_llm'])}")

        for s_name, s_func in STRATEGIES.items():
            scores = []
            y_scores = []
            current_y_rel = []
            current_y_str = []
            
            for it in item_stats:
                score = s_func(it["stats_lin"], it["stats_llm"])
                scores.append((it["cid"], score))
                y_scores.append(score)
                current_y_rel.append(it["y_rel"])
                current_y_str.append(it["y_str"])
            
            scores.sort(key=lambda x: -x[1])
            pred_ids = [cid for cid, _ in scores]
            rels = gold.get(qid, {})
            
            metrics[s_name]["p"].append(p_at_k(pred_ids, rels, 10, threshold=1))
            metrics[s_name]["ps"].append(p_at_k(pred_ids, rels, 10, threshold=2))
            metrics[s_name]["n"].append(ndcg_at_k(pred_ids, rels, 10))
            metrics[s_name]["a"].append(roc_auc(current_y_rel, y_scores))
            metrics[s_name]["as"].append(roc_auc(current_y_str, y_scores))

    print(f"Processed {processed} patients.")
    if processed == 0:
        print("No processed patients found. Please run run_framework.py first to generate results.")
        return

    print("-" * 80)
    print(f"{'Strategy':<25} {'P@10':<8} {'P@10(S)':<10} {'NDCG@10':<10} {'AUROC':<8} {'AUROC(S)':<10}")
    print("-" * 80)
    
    for s_name in STRATEGIES:
        m = metrics[s_name]
        if not m["p"]: continue
        
        avg_p = sum(m["p"])/len(m["p"])
        avg_ps = sum(m["ps"])/len(m["ps"])
        avg_n = sum(m["n"])/len(m["n"])
        avg_a = sum(m["a"])/len(m["a"])
        avg_as = sum(m["as"])/len(m["as"])
        
        print(f"{s_name:<25} {avg_p:.4f}   {avg_ps:.4f}     {avg_n:.4f}     {avg_a:.4f}   {avg_as:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="trec_2021")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    run_ablation(ds=args.ds, samples=args.samples)
