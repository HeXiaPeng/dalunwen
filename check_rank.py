import json
import os

path = "packages/match/data/TrialGPT/dataset/trec_2021/retrieved_trials.json"
with open(path, "r") as f:
    data = json.load(f)

for item in data:
    if item.get("patient_id") == "trec-20211":
        candidates = []
        for k, v in item.items():
            if isinstance(v, list):
                candidates.extend(v)
        
        print(f"Total candidates: {len(candidates)}")
        for i, c in enumerate(candidates):
            if c.get("NCTID") == "NCT00002569":
                print(f"Found NCT00002569 at rank {i} (0-indexed)")
                break
        else:
            print("NCT00002569 not found in candidates list")
        
        # Check rank of the first one in index.json: NCT01119599
        for i, c in enumerate(candidates):
            if c.get("NCTID") == "NCT01119599":
                print(f"Found NCT01119599 at rank {i} (0-indexed)")
                break
