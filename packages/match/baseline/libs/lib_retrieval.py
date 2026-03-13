import math
from collections import Counter

class BM25:
    def __init__(self, docs):
        self.docs=[]
        self.df={}
        self.avgdl=0.0
        for d in docs:
            tf = Counter(self.tok((d.get("title","")+" "+d.get("text","")).lower()))
            self.docs.append({"id": d["_id"], "tf": tf, "len": sum(tf.values())})
            for t in tf:
                self.df[t]=self.df.get(t,0)+1
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
