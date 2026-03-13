import os
import json
import re
import csv

AGE_RE = re.compile(r"(\d+)\s*[-]?\s*year[- ]?old", re.IGNORECASE)
GENDER_RE = re.compile(r"\b(male|female)\b", re.IGNORECASE)

def load_qrels(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            parts = line.strip().split()
            if first and parts and "query-id" in parts[0].lower():
                first = False
                continue
            if len(parts) < 3:
                parts = line.strip().split("\t")
            if len(parts) >= 3:
                qid, docid, rel = parts[0], parts[1], parts[2]
                try:
                    rel = int(rel)
                except:
                    rel = 1
                if rel > 0:
                    gold.setdefault(qid, set()).add(docid)
    return gold

def ie_rule_score(patient_text, inc_text, exc_text):
    age_match = AGE_RE.search(patient_text or "")
    age = int(age_match.group(1)) if age_match else None
    gender_match = GENDER_RE.search(patient_text or "")
    gender = gender_match.group(1).lower() if gender_match else None
    M_total = 0
    M_hit = 0
    R_vals = []
    # inclusion
    for crit in (inc_text or "").split("\n"):
        c = crit.lower()
        if "age" in c or "years" in c:
            M_total += 1
            if age is not None:
                # simple range detect
                m = re.findall(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*years", c)
                ok = None
                if m:
                    lo, hi = int(m[0][0]), int(m[0][1])
                    ok = (lo <= age <= hi)
                if ok is None:
                    # lower/upper bound
                    m1 = re.findall(r"age\s*(?:>=|≥|>)\s*(\d+)", c)
                    m2 = re.findall(r"age\s*(?:<=|≤|<)\s*(\d+)", c)
                    if m1:
                        ok = age >= int(m1[0])
                    if m2:
                        ok = (ok if ok is not None else True) and (age <= int(m2[0]))
                if ok:
                    M_hit += 1
                    R_vals.append(1.0)
                elif ok is False:
                    R_vals.append(0.0)
        if "male" in c or "female" in c or "both" in c or "all" in c:
            M_total += 1
            if gender is not None:
                if "both" in c or "all" in c:
                    M_hit += 1
                elif ("male" in c and gender == "male") or ("female" in c and gender == "female"):
                    M_hit += 1
    # exclusion: simple violation detect
    excl_safe = 0
    excl_total = 0
    for crit in (exc_text or "").split("\n"):
        c = crit.lower()
        violated = False
        if "male" in c and gender == "female":
            violated = True
        if "female" in c and gender == "male":
            violated = True
        # crude age violation
        if "age" in c or "years" in c:
            excl_total += 1
            if age is not None:
                m = re.findall(r"(\d+)\s*(?:-|to|–)\s*(\d+)\s*years", c)
                if m:
                    lo, hi = int(m[0][0]), int(m[0][1])
                    if not (lo <= age <= hi):
                        violated = True
                m1 = re.findall(r"age\s*(?:>=|≥|>)\s*(\d+)", c)
                if m1 and age < int(m1[0]):
                    violated = True
                m2 = re.findall(r"age\s*(?:<=|≤|<)\s*(\d+)", c)
                if m2 and age > int(m2[0]):
                    violated = True
        if "male" in c or "female" in c:
            excl_total += 1
        excl_safe += (0 if violated else 1)
    M = (M_hit / M_total) if M_total > 0 else 0.0
    E = (excl_safe / excl_total) if excl_total > 0 else 1.0
    R = sum(R_vals)/len(R_vals) if R_vals else 0.0
    CCS = 0.5*M + 0.3*E + 0.2*R
    EPR = 1 if (M >= 0.6 and E >= 0.9) else 0
    EVR = 0 if E >= 1.0 else 1
    return round(M,3), round(E,3), round(R,3), round(CCS,3), EPR, EVR

def eval_trec(dataset_dir, k=10, samples=None, export_csv=None):
    # load retrieved trials
    with open(os.path.join(dataset_dir, "retrieved_trials.json"), "r", encoding="utf-8") as f:
        retrieved = json.load(f)
    # qrels
    gold = load_qrels(os.path.join(dataset_dir, "qrels", "test.tsv"))
    rows=[]
    total_prec=0.0
    total_hit=0
    total_ccs=0.0
    total_epr=0
    total_evr=0
    count=0
    for entry in retrieved:
        qid = entry.get("patient_id") or entry.get("qid") or ""
        plist = entry.get("0") or []
        if not plist:
            continue
        # cap samples
        if samples and count>=samples:
            break
        count+=1
        top = plist[:k]
        pred_ids = [p.get("NCTID") for p in top if p.get("NCTID")]
        gset = gold.get(qid, set())
        hits = sum(1 for pid in pred_ids if pid in gset)
        prec = hits/k if k else 0.0
        hit = 1 if hits>0 else 0
        total_prec += prec
        total_hit += hit
        # I/E rule on Top-1
        inc = top[0].get("inclusion_criteria","") if top else ""
        exc = top[0].get("exclusion_criteria","") if top else ""
        M,E,R,CCS,EPR,EVR = ie_rule_score(entry.get("patient",""), inc, exc)
        total_ccs += CCS
        total_epr += EPR
        total_evr += EVR
        rows.append({
            "query_id": qid,
            "p_at_k": round(prec,3),
            "hit_at_k": hit,
            "gold_count": len(gset),
            "top1_ccs": CCS,
            "top1_epr": EPR,
            "top1_evr": EVR
        })
    avg_p = total_prec/count if count else 0.0
    hit_rate = total_hit/count if count else 0.0
    avg_ccs = total_ccs/count if count else 0.0
    epr_rate = total_epr/count if count else 0.0
    evr_rate = total_evr/count if count else 0.0
    print(json.dumps({
        "dataset": os.path.basename(dataset_dir),
        "k": k,
        "samples": count,
        "avg_p_at_k": round(avg_p,3),
        "hit_rate_at_k": round(hit_rate,3),
        "top1_avg_ccs": round(avg_ccs,3),
        "top1_epr_rate": round(epr_rate,3),
        "top1_evr_rate": round(evr_rate,3),
    }, ensure_ascii=False, indent=2))
    if export_csv and rows:
        with open(export_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV exported: {export_csv}")

if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "data", "TrialGPT", "dataset")
    eval_trec(os.path.join(base, "trec_2021"), k=10, samples=20, export_csv=os.path.join(os.path.dirname(__file__), "results_trec2021_eval.csv"))
    eval_trec(os.path.join(base, "trec_2022"), k=10, samples=20, export_csv=os.path.join(os.path.dirname(__file__), "results_trec2022_eval.csv"))
