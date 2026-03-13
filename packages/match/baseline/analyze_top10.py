import json
import os
import csv

# Paths
results_path = "packages/match/baseline/results/output/trec-20211/index.json"
qrels_path = "packages/match/data/TrialGPT/dataset/trec_2021/qrels/test.tsv"

# Load Qrels
gold = {}
with open(qrels_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                if qid == "trec-20211":
                    gold[docid] = rel
            except ValueError:
                continue

# Load Results
with open(results_path, "r") as f:
    data = json.load(f)
    results = data.get("results", [])

# Sort by LLM Score (Relevance + Eligibility seems to be the fusion logic in lib_aggregate.py: fusion = lin + llm. 
# But let's look at how llm_aggregate calculates: rank_score = rel. Excl score = -elig.
# Wait, let's check lib_aggregate.py again to be sure about "LLM Score".
# In run_framework.py: rs_llm, _, _ = llm_aggregate(item).
# lib_aggregate.py says:
# def llm_aggregate(match_json):
#    rel = float(match_json.get("relevance",0))
#    elig = float(match_json.get("eligibility",0))
#    rank_score = rel
#    return rank_score, ...
# So LLM Score is just 'relevance'.

# Let's sort by 'relevance' (descending)
results.sort(key=lambda x: float(x.get("relevance", 0)), reverse=True)

print(f"Analysis for Patient: trec-20211")
print("-" * 80)
print(f"{'Rank':<4} | {'Trial ID':<12} | {'LLM Score':<10} | {'True Label':<10} | {'Reasoning Snippet'}")
print("-" * 80)

for i, res in enumerate(results[:10]):
    tid = res.get("trial_id")
    score = res.get("relevance")
    label = gold.get(tid, "N/A")
    
    # Get first inclusion evidence as snippet
    evidence = ""
    for crit in res.get("criteria", []):
        if crit.get("label") == "include":
            evidence = crit.get("evidence", "")[:100] + "..."
            break
    if not evidence:
         for crit in res.get("criteria", []):
            evidence = crit.get("evidence", "")[:100] + "..."
            break
            
    print(f"{i+1:<4} | {tid:<12} | {score:<10} | {label:<10} | {evidence}")
