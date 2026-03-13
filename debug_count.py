import json
import os
import sys

# Add path to libs
sys.path.append(os.path.join(os.getcwd(), "packages/match/baseline"))
from libs.lib_dataset import load_dataset

def check_counts():
    print("Checking index.json count...")
    index_path = "packages/match/baseline/results/output/trec-20211/index.json"
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            data = json.load(f)
            results = data.get("results", [])
            print(f"Count in index.json: {len(results)}")
            
            # Check for duplicates in index.json
            seen_ids = set()
            duplicates = 0
            for r in results:
                tid = r.get("trial_id")
                if tid in seen_ids:
                    duplicates += 1
                seen_ids.add(tid)
            print(f"Unique trials in index.json: {len(seen_ids)}")
            print(f"Duplicate trials in index.json: {duplicates}")
    else:
        print(f"File not found: {index_path}")

    print("\nChecking dataset loading...")
    # Load dataset using the library function
    corpus, id2doc, queries, gold, q2candidates = load_dataset("trec_2021")
    
    qid = "trec-20211"
    raw_cands = q2candidates.get(qid, [])
    print(f"Candidates for {qid} in q2candidates: {len(raw_cands)}")
    
    # Check how many are in id2doc
    present_in_doc = [cid for cid in raw_cands if cid in id2doc]
    print(f"Candidates present in id2doc: {len(present_in_doc)}")
    
    missing = [cid for cid in raw_cands if cid not in id2doc]
    if missing:
        print(f"Missing candidates (not in id2doc): {len(missing)}")
        print(f"First 5 missing: {missing[:5]}")
    
    # Check if there are duplicates in raw_cands
    print(f"Unique candidates in q2candidates: {len(set(raw_cands))}")

if __name__ == "__main__":
    check_counts()
