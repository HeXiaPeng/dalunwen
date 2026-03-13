import os
import json
import time
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from aliy_client import AliyClient
from libs.lib_dataset import load_dataset
from libs.lib_retrieval import BM25
from libs.lib_matching import qwen_match_batch, build_batches
from libs.lib_aggregate import linear_aggregate, llm_aggregate, fusion_score
from libs.lib_metrics import ndcg_at_k, p_at_k, roc_auc, bar

STATE_PATH = os.path.join(os.path.dirname(__file__), "results", "framework_state.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "framework_results.csv")

def load_state():
    if not os.path.exists(STATE_PATH): return {"done_qids":[]}
    with open(STATE_PATH,"r",encoding="utf-8") as f: return json.load(f)

def save_state(state):
    with open(STATE_PATH,"w",encoding="utf-8") as f: json.dump(state,f,ensure_ascii=False,indent=2)

def append_rows(rows, header):
    exists=os.path.exists(RESULTS_PATH)
    mode="a"
    # Check if we need to reset file due to header mismatch
    if exists:
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                r=csv.reader(f)
                h=next(r)
                if h!=header:
                    mode="w"
                    exists=False
        except:
            mode="w"
            exists=False
            
    with open(RESULTS_PATH,mode,newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        for r in rows: w.writerow(r)

# 拆分后的入口
def run(ds="trec_2021", samples=5, top_candidates=500, model="qwen-plus", force=False, workers=20):
    client=AliyClient()
    corpus, id2doc, queries, gold, q2candidates = load_dataset(ds)
    # bm25=BM25(corpus)
    state=load_state()
    # Add strict metrics
    header=["qid","strategy","p@10","p@10_strict","ndcg@10","auroc","auroc_strict"]
    total=min(samples,len(queries))
    done=state.get("done_qids",[])
    processed=0
    rows=[]
    
    print(f"Running for {total} patients with {workers} workers...")
    
    for q in queries[:samples]:
        qid=q["_id"]
        qtext=q["text"]
        
        base_dir = os.path.join(os.path.dirname(__file__), "results", "batches", ds, qid)
        combined_results = []
        
        # Check if already done (cache hit)
        if (not force) and qid in done:
            # Load from disk
            if os.path.exists(base_dir):
                for fname in os.listdir(base_dir):
                    if fname.endswith("_result.json"):
                        try:
                            with open(os.path.join(base_dir, fname), "r", encoding="utf-8") as f:
                                res = json.load(f)
                                if res and "results" in res:
                                    combined_results.extend(res["results"])
                        except:
                            pass
            if not combined_results:
                # Fallback: if no files found, maybe we should re-run?
                # For now, just print warning
                print(f"Warning: {qid} marked done but no results found.")
        else:
            # Run LLM
            # cands=bm25.search(qtext, top_k=top_candidates)
            raw_cands = q2candidates.get(qid, [])
            if not raw_cands:
                print(f"Warning: No candidates for {qid}")
                cands = []
            else:
                limit = min(len(raw_cands), top_candidates)
                target_ids = raw_cands[:limit]
                cands = [(cid, 0.0) for cid in target_ids if cid in id2doc]
            cands_count = len(cands)
            patient_idx = processed + 1
            
            # build batches
            # use safer chunk size to avoid API timeouts and context limits
            chunk_size = 1
            batch_ids, batches = build_batches(id2doc, cands, qtext, chunk=chunk_size, qid=qid, base_dir=base_dir)
            futures={}
            call_total=len(batches)
            print(f"Patient {patient_idx}/{total} (Candidates: {cands_count}) {bar(0, call_total)}", end="\r", flush=True)
            
            with ThreadPoolExecutor(max_workers=workers) as ex:
                # Submit all tasks
                future_to_bid = {ex.submit(qwen_match_batch, client, model, qtext, trials_chunk): bid for bid, trials_chunk in zip(batch_ids, batches)}
                
                completed = 0
                last_print = time.time()
                
                for future in as_completed(future_to_bid):
                    bid = future_to_bid[future]
                    try:
                        res = future.result()
                    except Exception as e:
                        print(f"Error in batch {bid}: {e}")
                        res = {"results": []}
                    
                    # Immediately write to disk and release memory
                    out_path = os.path.join(base_dir, f"{bid}_result.json")
                    if res is None:
                        with open(out_path, "w", encoding="utf-8") as f: f.write("{}")
                    else:
                        with open(out_path, "w", encoding="utf-8") as f: json.dump(res, f, ensure_ascii=False, indent=2)
                    
                    # Do NOT append to combined_results here to save memory
                    del res
                    
                    completed += 1
                    now = time.time()
                    if now - last_print >= 30 or completed == call_total:
                        print(f"Patient {patient_idx}/{total} (Candidates: {cands_count}) {bar(completed, call_total)}", flush=True)
                        # print("Batches:", ",".join(batch_ids), flush=True)
                        last_print = now
            
            # Reload results from disk for metric calculation (after memory heavy lifting is done)
            if os.path.exists(base_dir):
                for fname in os.listdir(base_dir):
                    if fname.endswith("_result.json"):
                        try:
                            with open(os.path.join(base_dir, fname), "r", encoding="utf-8") as f:
                                res = json.load(f)
                                if res and "results" in res:
                                    combined_results.extend(res["results"])
                        except:
                            pass
            
            # Mark as done
            if qid not in done:
                done.append(qid)
                save_state({"done_qids": done})
            processed+=1
            print()

        # Process Results (Metric Calculation) - Runs for both cached and new
        if not combined_results:
            continue

        scored={"linear":[], "llm":[], "fusion":[]}
        y_true_relaxed=[]; y_true_strict=[]
        y_score_lin=[]; y_score_llm=[]; y_score_fus=[]
        
        for item in combined_results:
            cid=item.get("trial_id")
            rs_lin, xs_lin, stats_lin = linear_aggregate(item)
            rs_llm, xs_llm, stats_llm = llm_aggregate(item)
            rs_fus = fusion_score(rs_lin, rs_llm)
            scored["linear"].append((cid, rs_lin))
            scored["llm"].append((cid, rs_llm))
            scored["fusion"].append((cid, rs_fus))
            
            gt = gold.get(qid, {}).get(cid, 0)
            y_true_relaxed.append(1 if gt>=1 else 0)
            y_true_strict.append(1 if gt>=2 else 0)
            
            y_score_lin.append(rs_lin)
            y_score_llm.append(rs_llm)
            y_score_fus.append(rs_fus)

        # save aggregated batch outputs per user (patient) under baseline/output/<ds>/<qid>/index.json
        # Only needed if we just ran LLM? Or always? Always is good for debugging.
        output_dir = os.path.join(os.path.dirname(__file__), "results", "output", ds, qid)
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump({
                "qid": qid,
                "patient_text": qtext,
                "results": combined_results
            }, f, ensure_ascii=False, indent=2)
        
        for k in scored:
            scored[k].sort(key=lambda x: -x[1])
        
        pred_lin=[cid for cid,_ in scored["linear"]]
        pred_llm=[cid for cid,_ in scored["llm"]]
        pred_fus=[cid for cid,_ in scored["fusion"]]
        
        rels=gold.get(qid,{})
        
        rows.extend([
            {
                "qid": qid, "strategy":"linear", 
                "p@10": round(p_at_k(pred_lin, rels, 10, threshold=1),3),
                "p@10_strict": round(p_at_k(pred_lin, rels, 10, threshold=2),3),
                "ndcg@10": round(ndcg_at_k(pred_lin, rels, 10),3), 
                "auroc": round(roc_auc(y_true_relaxed, y_score_lin),3),
                "auroc_strict": round(roc_auc(y_true_strict, y_score_lin),3)
            },
            {
                "qid": qid, "strategy":"llm",    
                "p@10": round(p_at_k(pred_llm, rels, 10, threshold=1),3),
                "p@10_strict": round(p_at_k(pred_llm, rels, 10, threshold=2),3),
                "ndcg@10": round(ndcg_at_k(pred_llm, rels, 10),3), 
                "auroc": round(roc_auc(y_true_relaxed, y_score_llm),3),
                "auroc_strict": round(roc_auc(y_true_strict, y_score_llm),3)
            },
            {
                "qid": qid, "strategy":"fusion", 
                "p@10": round(p_at_k(pred_fus, rels, 10, threshold=1),3),
                "p@10_strict": round(p_at_k(pred_fus, rels, 10, threshold=2),3),
                "ndcg@10": round(ndcg_at_k(pred_fus, rels, 10),3), 
                "auroc": round(roc_auc(y_true_relaxed, y_score_fus),3),
                "auroc_strict": round(roc_auc(y_true_strict, y_score_fus),3)
            },
        ])
        
        append_rows(rows, header)
        rows=[]
        print(f"Processed metrics for {qid}")

    print(f"Finished processing {total} patients.")
    
    # Summary
    if os.path.exists(RESULTS_PATH):
        agg={}
        with open(RESULTS_PATH,"r",encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r:
                s=row["strategy"]
                agg.setdefault(s, {"p":[], "ps":[], "n":[], "a":[], "as":[]})
                agg[s]["p"].append(float(row.get("p@10",0)))
                agg[s]["ps"].append(float(row.get("p@10_strict",0)))
                agg[s]["n"].append(float(row.get("ndcg@10",0)))
                agg[s]["a"].append(float(row.get("auroc",0)))
                agg[s]["as"].append(float(row.get("auroc_strict",0)))
        out={}
        for s,v in agg.items():
            out[s]={
                "avg_p@10": round(sum(v["p"])/len(v["p"]),3) if v["p"] else 0.0,
                "avg_p@10_strict": round(sum(v["ps"])/len(v["ps"]),3) if v["ps"] else 0.0,
                "avg_ndcg@10": round(sum(v["n"])/len(v["n"]),3) if v["n"] else 0.0,
                "avg_auroc": round(sum(v["a"])/len(v["a"]),3) if v["a"] else 0.0,
                "avg_auroc_strict": round(sum(v["as"])/len(v["as"]),3) if v["as"] else 0.0,
            }
        print(json.dumps({"summary": out}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="trec_2021")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--top_candidates", type=int, default=500)
    parser.add_argument("--model", type=str, default="qwen-plus")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    run(ds=args.ds, samples=args.samples, top_candidates=args.top_candidates, model=args.model, force=args.force, workers=args.workers)
