import json
import os
import math
from collections import Counter, defaultdict

class SimpleBM25:
    def __init__(self, k1=1.2, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq = defaultdict(int)
        self.docs = []
        self.avgdl = 0.0
    @staticmethod
    def tokenize(text: str):
        text = (text or "").lower()
        tokens = []
        word = []
        for ch in text:
            if ch.isalnum():
                word.append(ch)
            else:
                if word:
                    tokens.append("".join(word))
                    word = []
        if word:
            tokens.append("".join(word))
        return tokens
    def add_document(self, doc_id: str, text: str):
        tokens = self.tokenize(text)
        tf = Counter(tokens)
        self.docs.append({"id": doc_id, "tf": tf, "len": len(tokens)})
        for term in tf.keys():
            self.doc_freq[term] += 1
    def consolidate(self):
        self.avgdl = (sum(d["len"] for d in self.docs) / len(self.docs)) if self.docs else 0.0
    def score(self, query_text: str, top_k: int = 10):
        q_tokens = self.tokenize(query_text)
        q_tf = Counter(q_tokens)
        N = len(self.docs)
        scores = []
        for d in self.docs:
            score = 0.0
            dl = d["len"] or 1
            for term, qcount in q_tf.items():
                df = self.doc_freq.get(term, 0)
                if df == 0:
                    continue
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                f = d["tf"].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                term_score = idf * (f * (self.k1 + 1)) / denom
                score += term_score
            scores.append((d["id"], score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def load_mock_trials(path):
    trials = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trials.append(json.loads(line))
    return trials

def is_recruiting_ok(study_status, only_recruiting=True):
    if not only_recruiting:
        return True
    return str(study_status).lower() in {
        "recruiting","not yet recruiting","enrolling by invitation"
    }

def gender_ok(trial_gender, patient_gender):
    if not patient_gender:
        return True
    tg = str(trial_gender or "").lower()
    if tg in {"both","all"}:
        return True
    return tg == str(patient_gender).lower()

def age_ok(trial_age, patient_age):
    if patient_age is None:
        return True
    ta = str(trial_age or "")
    low = None
    high = None
    if "years" in ta.lower():
        parts = ta.lower().replace("years","").replace("year","").replace("to","-")
        parts = [p.strip() for p in parts.split("-") if p.strip()]
        if len(parts) == 2:
            try:
                low = int(parts[0])
                high = int(parts[1])
            except:
                low = None
                high = None
    if ta.upper() in {"CHILD","ADULT","OLDER_ADULT"}:
        if ta.upper()=="CHILD":
            return patient_age < 18
        if ta.upper()=="ADULT":
            return 18 <= patient_age < 65
        if ta.upper()=="OLDER_ADULT":
            return patient_age >= 65
    if low is not None and high is not None:
        return low <= patient_age <= high
    return True

def ie_rule_score(patient_age, patient_gender, trial_age, trial_gender, recruiting_pass):
    must_total = 0
    must_hit = 0
    ta = str(trial_age or "")
    tg = str(trial_gender or "").lower()
    if ta.upper() in {"CHILD","ADULT","OLDER_ADULT"} or "years" in ta.lower():
        must_total += 1
        if age_ok(trial_age, patient_age):
            must_hit += 1
    if tg not in {"both","all",""}:
        must_total += 1
        if gender_ok(trial_gender, patient_gender):
            must_hit += 1
    M = (must_hit / must_total) if must_total > 0 else 0.0
    E = 1.0 if recruiting_pass else 0.0
    R = 1.0 if ("years" in ta.lower() and age_ok(trial_age, patient_age)) else 0.0
    CCS = round(0.5*M + 0.3*E + 0.2*R, 3)
    epr = 1 if (M >= 0.6 and E >= 0.9) else 0
    evr = 0 if E >= 1.0 else 1
    return M, E, R, CCS, epr, evr

def main():
    base = os.path.join(os.path.dirname(__file__), "mock_iface_trials.jsonl")
    trials = load_mock_trials(base)
    patient = {
        "age": 58,
        "gender": "Female",
        "isRecruiting": True,
        "query": "A 58-year-old woman with episodic chest pain, suspected coronary artery disease"
    }
    pool = []
    for t in trials:
        recruiting_pass = is_recruiting_ok(t.get("study_status"), patient["isRecruiting"])
        if not recruiting_pass:
            continue
        if not gender_ok(t.get("sex"), patient["gender"]):
            continue
        if not age_ok(t.get("age"), patient["age"]):
            continue
        M,E,R,CCS,epr,evr = ie_rule_score(patient["age"], patient["gender"], t.get("age"), t.get("sex"), recruiting_pass)
        tt = dict(t)
        tt["_ie"] = {"M": round(M,3), "E": round(E,3), "R": round(R,3), "CCS": CCS, "EPR": epr, "EVR": evr}
        pool.append(tt)
    engine = SimpleBM25()
    idmap = {}
    for t in pool:
        idmap[t["trial_id"]] = t
        text = " ".join([t.get("study_title",""), t.get("conditions","")])
        engine.add_document(t["trial_id"], text)
    engine.consolidate()
    topk = engine.score(patient["query"], top_k=5)
    results = []
    for rank,(tid,score) in enumerate(topk, start=1):
        doc = idmap.get(tid, {})
        results.append({
            "rank": rank,
            "trial_id": tid,
            "score": round(score,3),
            "title": doc.get("study_title",""),
            "url": doc.get("study_url",""),
            "conditions": doc.get("conditions",""),
            "phases": doc.get("phases",""),
            "study_status": doc.get("study_status",""),
            "ie": doc.get("_ie", {})
        })
    print(json.dumps({"query": patient, "results": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
