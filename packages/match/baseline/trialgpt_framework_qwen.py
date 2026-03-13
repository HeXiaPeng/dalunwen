import os
import json
import math
import csv
import time
from collections import defaultdict, Counter
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aliy_client import AliyClient
from term_normalizer import load_synonyms, normalize_text, truncate_text
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_ROOT = os.path.join(ROOT, "data", "TrialGPT", "dataset")
STATE_PATH = os.path.join(os.path.dirname(__file__), "framework_state.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "framework_results.csv")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            yield json.loads(line)

def load_dataset(ds):
    syn2canon = load_synonyms(os.path.join(ROOT, "resources", "synonyms.json"))
    corpus_path = os.path.join(DATA_ROOT, ds, "corpus.jsonl")
    norm_corpus=[]; id2doc={}; norm_queries=[]
    if os.path.exists(corpus_path):
        corpus = list(load_jsonl(corpus_path))
        for d in corpus:
            cid = d["_id"]
            title = normalize_text(d.get("title",""), syn2canon)
            text = normalize_text(d.get("text",""), syn2canon)
            text = truncate_text(text, max_chars=500)
            nd = {"_id": cid, "title": title, "text": text}
            norm_corpus.append(nd)
            id2doc[cid]=nd
        queries = list(load_jsonl(os.path.join(DATA_ROOT, ds, "queries.jsonl")))
        for q in queries:
            norm_queries.append({"_id": q["_id"], "text": normalize_text(q.get("text",""), syn2canon)})
    else:
        rt_path = os.path.join(DATA_ROOT, ds, "retrieved_trials.json")
        rt = []
        with open(rt_path, "r", encoding="utf-8") as f:
            rt = json.load(f)
        seen={}
        for e in rt:
            qid = e.get("patient_id") or e.get("_id")
            qtext = normalize_text(e.get("patient",""), syn2canon)
            norm_queries.append({"_id": qid, "text": qtext})
            lst = e.get("0") or []
            for t in lst:
                cid = t.get("NCTID")
                if not cid or cid in seen: continue
                title = normalize_text(t.get("brief_title",""), syn2canon)
                inc = normalize_text(t.get("inclusion_criteria",""), syn2canon)
                exc = normalize_text(t.get("exclusion_criteria",""), syn2canon)
                summ = normalize_text(t.get("brief_summary",""), syn2canon)
                text = f"Inclusion criteria: {inc}\nExclusion criteria: {exc}\nSummary: {summ}"
                text = truncate_text(text, max_chars=500)
                nd = {"_id": cid, "title": title, "text": text}
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
                first=False; continue
            if len(parts)<3:
                parts = line.strip().split("\t")
            if len(parts)>=3:
                qid, docid, rel = parts[0], parts[1], parts[2]
                try: rel=int(rel)
                except: rel=1
                gold[str(qid)][str(docid)]=rel
    return norm_corpus, id2doc, norm_queries, gold

class BM25:
    def __init__(self, docs):
        self.docs=[]
        self.df=defaultdict(int)
        self.avgdl=0.0
        for d in docs:
            tf = Counter(self.tok((d.get("title","")+" "+d.get("text","")).lower()))
            self.docs.append({"id": d["_id"], "tf": tf, "len": sum(tf.values())})
            for t in tf: self.df[t]+=1
        self.avgdl = sum(d["len"] for d in self.docs)/len(self.docs) if self.docs else 0.0
    @staticmethod
    def tok(s):
        out=[]; w=[]
        for ch in s:
            if ch.isalnum(): w.append(ch)
            else:
                if w: out.append("".join(w)); w=[]
        if w: out.append("".join(w))
        return out
    def search(self, q, top_k=500, k1=1.2, b=0.75):
        qtf = Counter(self.tok((q or "").lower()))
        N = len(self.docs)
        scores=[]
        for d in self.docs:
            s=0.0; dl=d["len"] or 1
            for term in qtf:
                df=self.df.get(term,0)
                if df==0: continue
                idf=math.log((N-df+0.5)/(df+0.5)+1)
                f=d["tf"].get(term,0)
                if f==0: continue
                denom = f + k1*(1-b+b*dl/(self.avgdl or 1))
                s += idf*(f*(k1+1))/denom
            scores.append((d["id"], s))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

def extract_criteria(text, max_inc=30, max_exc=30):
    t = text or ""
    low = t.lower()
    inc_idx = low.find("inclusion criteria")
    exc_idx = low.find("exclusion criteria")
    inc_block = t[inc_idx:(exc_idx if exc_idx!=-1 else len(t))] if inc_idx!=-1 else ""
    exc_block = t[exc_idx:] if exc_idx!=-1 else ""
    inc_lines = [l.strip() for l in inc_block.split("\n") if l.strip()]
    exc_lines = [l.strip() for l in exc_block.split("\n") if l.strip()]
    return inc_lines[:max_inc], exc_lines[:max_exc]

def qwen_match(client, model, patient_text, trial, inc_lines, exc_lines):
    crits=[]
    for l in inc_lines:
        if len(l)>3: crits.append({"type":"inclusion","text":l})
    for l in exc_lines:
        if len(l)>3: crits.append({"type":"exclusion","text":l})
    sys = "Return STRICT JSON with per-criterion label include|exclude|not_relevant and evidence; also trial-level relevance(0..100) and eligibility(-100..100) with |eligibility|<=relevance."
    payload = {"patient": patient_text[:1200], "trial_id": trial["_id"], "title": trial.get("title",""), "summary": trial.get("text","")[:400], "criteria": crits}
    messages=[{"role":"system","content":sys},{"role":"user","content":json.dumps(payload)}]
    content = client.chat_completions(messages=messages, model=model)
    content = str(content).replace("```json","").replace("```","").strip()
    try:
        return json.loads(content)
    except:
        return None

def qwen_match_batch(client, model, patient_text, batch_trials):
    sys = "Return STRICT JSON with list 'results' each containing trial_id, per-criterion labels include|exclude|not_relevant with evidence; and trial-level relevance(0..100) and eligibility(-100..100) with |eligibility|<=relevance."
    payload = {"patient": patient_text[:1200], "trials": batch_trials}
    messages=[{"role":"system","content":sys},{"role":"user","content":json.dumps(payload)}]
    content = client.chat_completions(messages=messages, model=model)
    content = str(content).replace("```json","").replace("```","").strip()
    try:
        return json.loads(content)
    except:
        return None

def linear_aggregate(match_json):
    inc=0; exc=0; nin=0; nex=0
    for c in match_json.get("criteria",[]):
        lab=(c.get("label") or "").lower()
        if c.get("type")=="inclusion":
            if lab=="include": inc+=1
            elif lab=="not_included" or lab=="not included": nin+=1
        elif c.get("type")=="exclusion":
            if lab=="exclude": exc+=1
            elif lab=="not_excluded" or lab=="not excluded": nex+=1
    total_in = inc + nin
    total_ex = exc + nex
    p_inc = (inc/total_in) if total_in>0 else 0.0
    p_nin = (nin/total_in) if total_in>0 else 0.0
    p_exc = (exc/total_ex) if total_ex>0 else 0.0
    p_nex = (nex/total_ex) if total_ex>0 else 0.0
    rank_score = (+p_inc) + (-p_nin) + (-p_exc) + (+p_nex)
    excl_score = (-p_inc) + (+p_nin) + (+p_exc) + (-p_nex)
    return rank_score, excl_score, {"p_inc":p_inc,"p_nin":p_nin,"p_exc":p_exc,"p_nex":p_nex}

def llm_aggregate(match_json):
    rel = float(match_json.get("relevance",0))
    elig = float(match_json.get("eligibility",0))
    rank_score = rel
    excl_score = -elig
    return rank_score, excl_score, {"rel":rel, "elig":elig}

def fusion_score(stats_lin, stats_llm):
    fused = stats_lin["p_inc"] - (1 if stats_lin["p_nin"]>0 else 0) + stats_llm["rel"]
    return fused

def ndcg_at_k(pred_ids, rels, k=10):
    def dcg(items):
        s=0.0
        for i, docid in enumerate(items[:k], start=1):
            gain = rels.get(docid, 0)
            s += ((2**gain -1))/math.log2(i+1)
        return s
    ideal = sorted(rels.items(), key=lambda x: -x[1])
    ideal_ids = [docid for docid,_ in ideal][:k]
    idcg = dcg(ideal_ids) or 1.0
    return dcg(pred_ids)/idcg

def p_at_k(pred_ids, rels, k=10):
    hits = sum(1 for d in pred_ids[:k] if rels.get(d,0)>=1)
    return hits/k

def roc_auc(y_true, y_score):
    pairs=0; better=0; ties=0
    pos=[y_score[i] for i in range(len(y_true)) if y_true[i]==1]
    neg=[y_score[i] for i in range(len(y_true)) if y_true[i]==0]
    for ps in pos:
        for ns in neg:
            pairs+=1
            if ps>ns: better+=1
            elif ps==ns: ties+=1
    return (better + 0.5*ties)/pairs if pairs>0 else 0.0

def load_state():
    if not os.path.exists(STATE_PATH): return {"done_qids":[]}
    with open(STATE_PATH,"r",encoding="utf-8") as f: return json.load(f)

def save_state(state):
    with open(STATE_PATH,"w",encoding="utf-8") as f: json.dump(state,f,ensure_ascii=False,indent=2)

def append_rows(rows, header):
    exists=os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        for r in rows: w.writerow(r)

def run(ds="sigir", samples=5, top_candidates=50, model="qwen-plus"):
    client=AliyClient()
    corpus, id2doc, queries, gold = load_dataset(ds)
    bm25=BM25(corpus)
    state=load_state()
    header=["qid","strategy","p@10","ndcg@10","auroc"]
    total=min(samples,len(queries))
    done=state.get("done_qids",[])
    processed=0
    rows=[]
    def bar(current, total, width=24):
        total = max(total, 1)
        filled = int(width * current / total)
        return "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {current}/{total}"
    for q in queries[:samples]:
        qid=q["_id"]
        if qid in done:
            continue
        qtext=q["text"]
        cands=bm25.search(qtext, top_k=top_candidates)
        patient_idx = processed + 1
        scored={"linear":[], "llm":[], "fusion":[]}
        y_true=[]; y_score_lin=[]; y_score_llm=[]; y_score_fus=[]
        batch_items=[]
        for cid, base_s in cands:
            trial=id2doc[cid]
            inc, exc = extract_criteria(trial.get("text",""))
            batch_items.append({
                "trial_id": cid,
                "title": trial.get("title",""),
                "summary": trial.get("text","")[:400],
                "criteria": [{"type":"inclusion","text":l} for l in inc[:15]] + [{"type":"exclusion","text":l} for l in exc[:15]]
            })
        batch_size = 25
        batches = [batch_items[i:i+batch_size] for i in range(0, len(batch_items), batch_size)]
        batch_ids = [f"batch-{qid}-{i}-{int(time.time())}" for i in range(len(batches))]
        futures = {}
        call_total = len(batches)
        print(f"Patient {patient_idx}/{total} {bar(0, call_total)}", end="\r", flush=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "batches"), exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as ex:
            for bid, trials_chunk in zip(batch_ids, batches):
                in_path = os.path.join(os.path.dirname(__file__), "batches", f"{bid}.json")
                with open(in_path, "w", encoding="utf-8") as f:
                    json.dump({"patient": qtext, "trials": trials_chunk}, f, ensure_ascii=False, indent=2)
                futures[bid] = ex.submit(qwen_match_batch, client, model, qtext, trials_chunk)
            completed = 0
            last_print = time.time()
            while completed < call_total:
                completed = sum(1 for f in futures.values() if f.done())
                now = time.time()
                if now - last_print >= 60 or completed == call_total:
                    print(f"Patient {patient_idx}/{total} {bar(completed, call_total)}", flush=True)
                    last_print = now
                time.sleep(1)
        for bid, fut in futures.items():
            res = fut.result()
            out_path = os.path.join(os.path.dirname(__file__), "batches", f"{bid}_result.json")
            if res is None:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("{}")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            for item in res.get("results", []):
                cid = item.get("trial_id")
                rs_lin, xs_lin, stats_lin = linear_aggregate(item)
                rs_llm, xs_llm, stats_llm = llm_aggregate(item)
                rs_fus = fusion_score(stats_lin, stats_llm)
                scored["linear"].append((cid, rs_lin))
                scored["llm"].append((cid, rs_llm))
                scored["fusion"].append((cid, rs_fus))
                gt = gold.get(qid, {}).get(cid, 0)
                y_true.append(1 if gt>=1 else 0)
                y_score_lin.append(xs_lin)
                y_score_llm.append(xs_llm)
                y_score_fus.append(rs_fus)
        print()
        for k in scored:
            scored[k].sort(key=lambda x: -x[1])
        pred_lin = [cid for cid,_ in scored["linear"]]
        pred_llm = [cid for cid,_ in scored["llm"]]
        pred_fus = [cid for cid,_ in scored["fusion"]]
        rels = gold.get(qid, {})
        rows.extend([
            {"qid": qid, "strategy":"linear", "p@10": round(p_at_k(pred_lin, rels, 10),3), "ndcg@10": round(ndcg_at_k(pred_lin, rels, 10),3), "auroc": round(roc_auc(y_true, y_score_lin),3)},
            {"qid": qid, "strategy":"llm",    "p@10": round(p_at_k(pred_llm, rels, 10),3), "ndcg@10": round(ndcg_at_k(pred_llm, rels, 10),3), "auroc": round(roc_auc(y_true, y_score_llm),3)},
            {"qid": qid, "strategy":"fusion", "p@10": round(p_at_k(pred_fus, rels, 10),3), "ndcg@10": round(ndcg_at_k(pred_fus, rels, 10),3), "auroc": round(roc_auc(y_true, y_score_fus),3)},
        ])
        append_rows(rows, header)
        rows=[]
        done.append(qid)
        save_state({"done_qids": done})
        processed += 1
        print(f"{processed}/{total}")
    print(f"Processed {processed}/{total}")
    # summary
    if os.path.exists(RESULTS_PATH):
        agg={}
        with open(RESULTS_PATH,"r",encoding="utf-8") as f:
            r=csv.DictReader(f)
            for row in r:
                s=row["strategy"]
                agg.setdefault(s, {"p":[], "n":[], "a":[]})
                agg[s]["p"].append(float(row["p@10"]))
                agg[s]["n"].append(float(row["ndcg@10"]))
                agg[s]["a"].append(float(row["auroc"]))
        out={}
        for s,v in agg.items():
            out[s]={"avg_p@10": round(sum(v["p"])/len(v["p"]),3) if v["p"] else 0.0,
                    "avg_ndcg@10": round(sum(v["n"])/len(v["n"]),3) if v["n"] else 0.0,
                    "avg_auroc": round(sum(v["a"])/len(v["a"]),3) if v["a"] else 0.0,
                    "overall_avg": round(((sum(v["p"])/len(v["p"])) + (sum(v["n"])/len(v["n"])) + (sum(v["a"])/len(v["a"])))/3,3) if v["p"] and v["n"] and v["a"] else 0.0}
        print(json.dumps({"summary": out}, ensure_ascii=False, indent=2))

if __name__=="__main__":
    run(ds="trec_2021", samples=5, top_candidates=500, model="qwen-plus")
