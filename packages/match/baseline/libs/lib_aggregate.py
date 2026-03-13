def _norm_label(s):
    s = (s or "").lower().strip()
    s = s.replace(" ", "_")
    if s in ("included",): return "include"
    if s in ("not_included", "notincluded"): return "not_included"
    if s in ("excluded",): return "exclude"
    if s in ("not_excluded","notexcluded"): return "not_excluded"
    if s in ("not_applicable","na"): return "not_applicable"
    if s in ("not_enough_information","nei","no_info"): return "not_enough_information"
    if s in ("not_relevant","irrelevant"): return "not_relevant"
    return s

def linear_aggregate(match_json):
    # TrialGPT-like Hard Penalty Logic
    inc=0; exc=0; nin=0; nex=0; nei=0
    
    # Count criteria results
    for c in match_json.get("criteria",[]):
        lab=_norm_label(c.get("label"))
        t=c.get("type")
        
        # Fallback for when type is missing (e.g. baseline LLM output)
        if not t:
            if lab in ("include", "not_included"): t = "inclusion"
            elif lab in ("exclude", "not_excluded"): t = "exclusion"
            
        if t=="inclusion":
            if lab=="include": inc+=1
            elif lab=="not_included": nin+=1
            elif lab=="not_enough_information": nei+=1
        elif t=="exclusion":
            if lab=="exclude": exc+=1
            elif lab=="not_excluded": nex+=1
            elif lab=="not_enough_information": nei+=1
            
    # Calculate Base Score (Match Ratio)
    # TrialGPT: score += included / (included + not_inc + no_info_inc + eps)
    denom_inc = inc + nin + nei
    base_score = (inc / denom_inc) if denom_inc > 0 else 0.0
    
    # Apply Hard Penalties
    # TrialGPT: if not_inc > 0: score -= 1
    # TrialGPT: if excluded > 0: score -= 1
    rank_score = base_score
    if nin > 0:
        rank_score -= 1.0
    if exc > 0:
        rank_score -= 1.0
        
    # Exclusion score (inverse) for reference
    excl_score = -rank_score
    
    return rank_score, excl_score, {"inc":inc,"nin":nin,"exc":exc,"nex":nex,"nei":nei, "base": base_score}

def llm_aggregate(match_json):
    rel = float(match_json.get("relevance",0))
    elig = float(match_json.get("eligibility",0))
    rank_score = rel
    excl_score = -elig
    return rank_score, excl_score, {"rel":rel, "elig":elig}

def fusion_score(score_lin, score_llm):
    # Combine Hard Penalty Score with LLM Relevance
    # New logic: 
    # If Linear Score is penalized (<= -1), it means there is a HARD EXCLUSION.
    # In this case, we should NOT let LLM Relevance rescue it.
    # We should keep the score low.
    
    if score_lin <= -0.5:
        # Hard exclusion detected by Linear Logic.
        # Even if LLM gives 100 (1.0), the result should be negative or very low.
        # score_lin is like -1 or -2. 
        # return score_lin + 0.1 * (score_llm / 100.0) # Downweight LLM
        # Or just return score_lin?
        return score_lin
        
    # If no hard exclusion, we can add LLM bonus.
    # LLM score is 0-100. We normalize it to 0-1.
    return score_lin + (score_llm / 100.0)
