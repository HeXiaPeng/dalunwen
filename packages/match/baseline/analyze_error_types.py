import csv
import re
import os
from collections import Counter

def analyze_error_types(csv_path="baseline/results/error_analysis.csv"):
    # Fix path if running from root
    if not os.path.exists(csv_path):
        # Try full path
        csv_path = os.path.join(os.path.dirname(__file__), "results", "error_analysis.csv")
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Patterns for classification
    patterns = {
        "Inference_from_Silence": [
            r"no mention of", r"not stated", r"does not indicate", r"silent on", 
            r"no information", r"not reported", r"no evidence of", r"no indication",
            r"not described", r"denies", r"unremarkable"
        ],
        "Demographic_Match": [
            r"\d+[- ]years?[- ]old", r"years? of age", r"\bman\b", r"\bwoman\b",
            r"\bmale\b", r"\bfemale\b", r"gender", r"sex", r"age >", r"age <", r"age \d+"
        ],
        "Diagnosis_Match": [
            r"diagnosed with", r"history of", r"known case of", r"confirmed", 
            r"suffering from", r"presents with", r"has", r"patient is a .* with"
        ],
        "Lab_Value_Check": [
            r"level", r"mg/dl", r"mmol/l", r"g/dl", r"count", r"saturation", r"bp", r"hr", 
            r"bmi", r"weight", r"height"
        ]
    }

    stats = Counter()
    total = 0
    
    # Store examples for each category
    examples = {k: [] for k in patterns.keys()}
    examples["Other"] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            reason = row["model_reasoning"].lower()
            
            # Check matches
            matched_types = []
            for p_name, p_list in patterns.items():
                for p in p_list:
                    if re.search(p, reason):
                        matched_types.append(p_name)
                        break
            
            if not matched_types:
                stats["Other"] += 1
                examples["Other"].append(row)
            else:
                for mt in matched_types:
                    stats[mt] += 1
                    if len(examples[mt]) < 3: # Keep top 3 examples
                        examples[mt].append(row)

    # Print Report
    print(f"Total False Positives Analyzed: {total}")
    print("-" * 60)
    print(f"{'Error Category':<25} {'Count':<8} {'Percentage (of cases)':<25}")
    print("-" * 60)
    
    # Sort by count
    for cat, count in stats.most_common():
        pct = (count / total) * 100
        print(f"{cat:<25} {count:<8} {pct:.1f}%")
        
    print("-" * 60)
    print("\nExamples per Category (Top 3):")
    for cat in patterns.keys():
        rows = examples.get(cat, [])
        if not rows: continue
        print(f"\n[{cat}]")
        for i, r in enumerate(rows):
            # Clean up reasoning for display
            reason = r['model_reasoning'].replace("\n", " ")
            if len(reason) > 120: reason = reason[:120] + "..."
            print(f"  {i+1}. {r['trial_id']} (Score {r['model_score']}): {reason}")
            
    if examples["Other"]:
        print(f"\n[Other]")
        for i, r in enumerate(examples["Other"][:3]):
            reason = r['model_reasoning'].replace("\n", " ")
            if len(reason) > 120: reason = reason[:120] + "..."
            print(f"  {i+1}. {r['trial_id']} (Score {r['model_score']}): {reason}")

if __name__ == "__main__":
    analyze_error_types()
