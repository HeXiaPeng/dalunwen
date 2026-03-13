import os
import json
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

from libs.lib_dataset import load_dataset
from libs.lib_aggregate import linear_aggregate, llm_aggregate, fusion_score
from libs.lib_metrics import ndcg_at_k, p_at_k, roc_auc

# Adjust these paths to match your actual file locations
# The file user mentioned: packages/match/data/TrialGPT/results/matching_results_trec_2021_qwen-plus.json
# We will use this as input.

INPUT_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "TrialGPT", "results", "matching_results_trec_2021_qwen-plus.json")
OUTPUT_CSV_PATH = os.path.join(os.path.dirname(INPUT_JSON_PATH), "re_ranked_trialgpt_results.csv")

def re_rank(ds="trec_2021"):
    print(f"Loading dataset {ds} for ground truth...")
    # We only need gold standards here
    # load_dataset returns: corpus, id2doc, queries, gold, q2candidates
    # We only need 'gold' (4th item) and maybe 'queries' (3rd) if needed
    _, _, queries, gold, _ = load_dataset(ds)
    
    print(f"Loading TrialGPT results from {INPUT_JSON_PATH}...")
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"Error: File {INPUT_JSON_PATH} not found.")
        return

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        matching_results = json.load(f)

    header = ["qid", "strategy", "p@10", "p@10_strict", "ndcg@10", "auroc", "auroc_strict"]
    rows = []
    
    # Iterate over patients in the JSON
    # Structure: { "trec-20211": { "0": { "NCT...":Result, ... }, "1": ... } }
    
    # We need to flatten the batch keys ("0", "1", "2"...)
    
    print("Processing patients...")
    
    for qid, batches in matching_results.items():
        # Flatten all trials for this patient
        all_trials = {} # trial_id -> result_obj
        
        for batch_key, trial_map in batches.items():
            if not isinstance(trial_map, dict): continue
            for trial_id, res in trial_map.items():
                all_trials[trial_id] = res
                
        if not all_trials:
            continue
            
        # Calculate scores
        scored = {"linear": [], "llm": [], "fusion": []}
        y_true_relaxed = []
        y_true_strict = []
        y_score_lin = []
        y_score_llm = []
        y_score_fus = []
        
        # Note: TrialGPT result format in JSON might be slightly different from what our lib expects?
        # Let's check the format from user input:
        # "NCT...": { "inclusion": { "0": [text, evidence, label], ... }, "exclusion": ... }
        # This matches what linear_aggregate expects (it looks for "inclusion"/"exclusion" keys).
        # However, llm_aggregate expects "relevance" and "eligibility" fields.
        # The TrialGPT JSON shown by user DOES NOT contain "relevance" or "eligibility" scores directly in the matching block.
        # Those are usually in a separate "aggregation" file or step.
        # IF the JSON doesn't have them, llm_aggregate will return 0.
        
        for cid, item in all_trials.items():
            # Construct a wrapper if needed, or pass item directly
            # item has "inclusion", "exclusion"
            # It might NOT have "relevance", "eligibility".
            # If missing, LLM score will be 0. 
            
            # For linear_aggregate, we need to adapt the structure slightly if it differs.
            # Our lib expects: match_json.get("criteria", []) -> list of {label, type, ...}
            # BUT the TrialGPT JSON has: match_json["inclusion"]["0"] -> [text, evidence, label]
            # We need to convert TrialGPT format to our Lib format ON THE FLY.
            
            criteria_list = []
            
            # Process Inclusion
            inc_map = item.get("inclusion", {})
            if isinstance(inc_map, dict):
                for k, v in inc_map.items():
                    if isinstance(v, list) and len(v) >= 3:
                        label = v[2] # "included", "not included"
                        criteria_list.append({"type": "inclusion", "label": label})
            
            # Process Exclusion
            exc_map = item.get("exclusion", {})
            if isinstance(exc_map, dict):
                for k, v in exc_map.items():
                    if isinstance(v, list) and len(v) >= 3:
                        label = v[2] # "excluded", "not excluded"
                        criteria_list.append({"type": "exclusion", "label": label})
            
            # Create a fake object for our library
            adapter_obj = {
                "criteria": criteria_list,
                "relevance": item.get("relevance_score", 0), # Guessing key
                "eligibility": item.get("eligibility_score", 0) # Guessing key
            }
            
            # But wait, the user wants to re-score using OUR new "Hard Penalty" logic in linear_aggregate.
            # Our modified linear_aggregate uses `match_json.get("criteria",[])`.
            
            rs_lin, _, _ = linear_aggregate(adapter_obj)
            
            # LLM Score: The JSON shown doesn't have it. So LLM score will be 0.
            # Fusion will just be Linear Score.
            rs_llm = 0.0 
            rs_fus = fusion_score(rs_lin, rs_llm)
            
            scored["linear"].append((cid, rs_lin))
            scored["llm"].append((cid, rs_llm))
            scored["fusion"].append((cid, rs_fus))
            
            gt = gold.get(qid, {}).get(cid, 0)
            y_true_relaxed.append(1 if gt >= 1 else 0)
            y_true_strict.append(1 if gt >= 2 else 0)
            
            y_score_lin.append(rs_lin)
            y_score_llm.append(rs_llm)
            y_score_fus.append(rs_fus)
            
        # Sort and Metric
        for k in scored:
            scored[k].sort(key=lambda x: -x[1])
            
        pred_lin = [cid for cid, _ in scored["linear"]]
        # pred_llm is useless if all 0
        pred_fus = [cid for cid, _ in scored["fusion"]]
        
        rels = gold.get(qid, {})
        
        # We only care about Linear/Fusion (since LLM score is missing in this file)
        # Actually Fusion = Linear here.
        
        rows.append({
            "qid": qid, "strategy": "re-ranked-linear",
            "p@10": round(p_at_k(pred_lin, rels, 10, threshold=1), 3),
            "p@10_strict": round(p_at_k(pred_lin, rels, 10, threshold=2), 3),
            "ndcg@10": round(ndcg_at_k(pred_lin, rels, 10), 3),
            "auroc": round(roc_auc(y_true_relaxed, y_score_lin), 3),
            "auroc_strict": round(roc_auc(y_true_strict, y_score_lin), 3)
        })

    # Save
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
            
    print(f"Saved re-ranked results to {OUTPUT_CSV_PATH}")
    
    # Calc Average
    if rows:
        avg_p = sum(r["p@10"] for r in rows) / len(rows)
        avg_ps = sum(r["p@10_strict"] for r in rows) / len(rows)
        avg_n = sum(r["ndcg@10"] for r in rows) / len(rows)
        avg_a = sum(r["auroc"] for r in rows) / len(rows)
        avg_as = sum(r["auroc_strict"] for r in rows) / len(rows)
        
        print("\nSummary (Re-ranked Linear/Fusion):")
        print(f"Avg P@10 (Relaxed): {avg_p:.3f}")
        print(f"Avg P@10 (Strict):  {avg_ps:.3f}")
        print(f"Avg NDCG@10:        {avg_n:.3f}")
        print(f"Avg AUROC (Relaxed):{avg_a:.3f}")
        print(f"Avg AUROC (Strict): {avg_as:.3f}")

if __name__ == "__main__":
    re_rank()
