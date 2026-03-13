import os
import json
from collections import defaultdict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from term_normalizer import load_synonyms, normalize_text, truncate_text

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_ROOT = os.path.join(ROOT, "data", "TrialGPT", "dataset")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)

def load_dataset(ds):
    syn2canon = load_synonyms(os.path.join(ROOT, "resources", "synonyms.json"))
    corpus_path = os.path.join(DATA_ROOT, ds, "corpus.jsonl")
    norm_corpus=[]; id2doc={}; norm_queries=[]; q2candidates={}
    if os.path.exists(corpus_path):
        corpus = list(load_jsonl(corpus_path))
        for d in corpus:
            cid = d["_id"]
            title = normalize_text(d.get("title",""), syn2canon)
            # text = normalize_text(d.get("text",""), syn2canon)
            # Do NOT truncate heavily or normalize aggressively if we want to extract criteria later.
            # But the current pipeline relies on 'text' being the full content.
            # Let's keep the raw text as well?
            raw_text = d.get("text", "")
            # text = truncate_text(text, max_chars=500) # This truncation is killing the criteria!
            # We should increase max_chars significantly.
            nd = {"_id": cid, "title": title, "text": raw_text, "meta": d.get("metadata", {})} # Use raw text
            norm_corpus.append(nd)
            id2doc[cid]=nd
        queries = list(load_jsonl(os.path.join(DATA_ROOT, ds, "queries.jsonl")))
        for q in queries:
            norm_queries.append({"_id": q["_id"], "text": normalize_text(q.get("text",""), syn2canon)})
    else:
        # Priority 2: Load from retrieved_trials.json (Default for TrialGPT datasets)
        # This file contains grouped candidates under keys "0", "1", "2" (and potentially others).
        # We need to flatten them into a single list of candidates per patient.
        rt_path = os.path.join(DATA_ROOT, ds, "retrieved_trials.json")
        rt = []
        with open(rt_path, "r", encoding="utf-8") as f:
            rt = json.load(f)
        seen={}
        
        for e in rt:
            # 1. Extract Patient ID and Text
            qid = e.get("patient_id") or e.get("_id")
            # Use raw patient text without normalization to preserve medical details?
            # Or normalize? TrialGPT normalizes sentences. Let's keep it simple.
            qtext = normalize_text(e.get("patient",""), syn2canon)
            
            # Check if this query is already added to avoid duplicates
            if not any(q["_id"] == qid for q in norm_queries):
                norm_queries.append({"_id": qid, "text": qtext})
            
            # 2. Extract Candidates from all relevant keys ("0", "1", "2")
            q2candidates[qid]=[]
            candidate_list = []
            
            # Iterate over all keys in the patient object
            for k, v in e.items():
                # Skip metadata keys
                if k in ("patient_id", "patient", "_id"):
                    continue
                # If the value is a list (e.g., "0": [...], "1": [...]), add it
                if isinstance(v, list):
                    candidate_list.extend(v)
            
            # 3. Process each candidate trial
            for t in candidate_list:
                cid = t.get("NCTID")
                if not cid: continue
                
                # Add to candidate list for this query
                if cid not in q2candidates[qid]:
                    q2candidates[qid].append(cid)
                
                # Add to global document store (id2doc) if not seen
                if cid in seen:
                    continue
                
                title = normalize_text(t.get("brief_title",""), syn2canon)
                inc = t.get("inclusion_criteria","")
                exc = t.get("exclusion_criteria","")
                summ = t.get("brief_summary","")
                
                # Construct full text for matching
                text = f"Inclusion criteria: {inc}\nExclusion criteria: {exc}\nSummary: {summ}"
                
                # Store document
                nd = {
                    "_id": cid, 
                    "title": title, 
                    "text": text,
                    "brief_title": t.get("brief_title", ""),
                    "phase": t.get("phase", ""),
                    "drugs": t.get("drugs", ""),
                    "drugs_list": t.get("drugs_list", []),
                    "diseases": t.get("diseases", ""),
                    "diseases_list": t.get("diseases_list", []),
                    "enrollment": t.get("enrollment", ""),
                    "inclusion_criteria": inc,
                    "exclusion_criteria": exc,
                    "brief_summary": summ,
                    "NCTID": t.get("NCTID", "")
                }
                norm_corpus.append(nd)
                id2doc[cid]=nd
                seen[cid]=1
    gold = defaultdict(dict)
    qp = os.path.join(DATA_ROOT, ds, "qrels", "test.tsv")
    with open(qp, "r", encoding="utf-8") as f:
        first=True
        for line in f:
            parts = line.strip().split()
            if first and parts and "query-id" in parts[0].lower():
                first=False
                continue
            if len(parts)<3:
                parts = line.strip().split("\t")
            if len(parts)>=3:
                qid, docid, rel = parts[0], parts[1], parts[2]
                try:
                    rel=int(rel)
                except:
                    rel=1
                gold[str(qid)][str(docid)]=rel
    return norm_corpus, id2doc, norm_queries, gold, q2candidates
