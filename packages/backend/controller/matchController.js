const { Op } = require('sequelize');
const ClinicalTrialUsa = require('../model/ClinicalTrialUsa');
const ClinicalTrialChina = require('../model/ClinicalTrialChina');
const aiClient = require('../utils/aiClient');
const bm25 = require('wink-bm25-text-search');
const winkNlpUtils = require('wink-nlp-utils');
const { ALI_LLM_CONFIG } = require('../config/aiConfig');

function getRegistryConfig(registry) {
  const key = String(registry || 'usa').toLowerCase();
  if (key === 'china' || key === 'cn') {
    return {
      registry: 'china',
      model: ClinicalTrialChina,
      idField: 'reg_number',
      urlField: 'study_url',
    };
  }
  return {
    registry: 'usa',
    model: ClinicalTrialUsa,
    idField: 'nct_number',
    urlField: 'study_url',
  };
}

async function extractQueryAndKeywords(inputText) {
  const raw = String(inputText || '').trim();
  if (!raw) return { query: '', keywords: [] };

  const stopwords = new Set([
    'the','a','an','and','or','of','to','in','on','for','with','without','by','from','as','at','is','are','was','were',
    'be','been','being','this','that','these','those','it','its','he','she','they','them','his','her','their','patient',
    'years','year','old','male','female','over','under','age','has','have','had','not','no','yes','can','could','may','might',
    'will','would','should','must','also','into','than','then','when','while','which','who','whom','what','where','why','how',
    'full','capacity','conduct','history','scheduled','undergo','plan','treatment','diagnosed','diagnosis'
  ]);

  if (ALI_LLM_CONFIG.API_KEY) {
    const prompt = `
You will receive a long patient condition/description text. Convert it into search terms for retrieving relevant clinical trials.

Rules:
1) Output STRICT JSON only.
2) Provide a short query string (<= 120 characters).
3) Provide 6-12 keywords, prioritize diseases, organs, interventions, biomarkers, drug/radiotracer names.
4) Keywords should be short phrases.

Input:
${raw}

Output JSON format:
{ "query": "...", "keywords": ["...", "..."] }
`;

    try {
      const completion = await aiClient.chat.completions.create({
        model: ALI_LLM_CONFIG.MODEL,
        messages: [
          { role: 'system', content: 'You extract search queries and keywords. Output only valid JSON.' },
          { role: 'user', content: prompt },
        ],
        response_format: { type: 'json_object' },
      });

      const content = String(completion.choices?.[0]?.message?.content || '')
        .replace(/```json/g, '')
        .replace(/```/g, '')
        .trim();
      const parsed = JSON.parse(content || '{}');
      const query = String(parsed.query || '').trim();
      const keywords = Array.isArray(parsed.keywords)
        ? parsed.keywords.map((k) => String(k).trim()).filter((k) => k.length > 0)
        : [];
      return { query, keywords };
    } catch (_e) {
      return { query: '', keywords: [] };
    }
  }

  const tokens = winkNlpUtils.string
    .tokenize0(winkNlpUtils.string.removeExtraSpaces(winkNlpUtils.string.lowerCase(raw)))
    .map((t) => String(t).replace(/[^a-z0-9\-]/g, ''))
    .filter((t) => t.length >= 4 && !stopwords.has(t));

  const freq = new Map();
  tokens.forEach((t) => freq.set(t, (freq.get(t) || 0) + 1));

  const keywords = Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([t]) => t);

  const query = keywords.slice(0, 6).join(' ');
  return { query, keywords };
}

async function matchPatient(ctx) {
  try {
    const { age, gender, condition, isRecruiting, registry } = ctx.request.body;
    const registryConfig = getRegistryConfig(registry);

    if (!ALI_LLM_CONFIG.API_KEY) {
      ctx.status = 500;
      ctx.body = { code: 500, msg: 'Missing LLM API key. Set ALI_LLM_API_KEY or DASHSCOPE_API_KEY.' };
      return;
    }

    // 1. Coarse Screening (Database Filter)
    const whereClause = {};

    // Recruitment Status Filter
    if (isRecruiting) {
      whereClause.study_status = {
        [Op.in]: [
          'RECRUITING',
          'Recruiting',
          'NOT_YET_RECRUITING',
          'Not yet recruiting',
          'ENROLLING_BY_INVITATION',
          'Enrolling by invitation',
        ],
      };
    }

    // Gender Filter
    if (gender) {
      const g = gender.toUpperCase(); 
      const targetGender = g === 'MALE' ? ['Male', 'MALE'] : ['Female', 'FEMALE'];
      const universal = ['All', 'ALL', 'Both', 'BOTH'];
      
      whereClause.sex = {
        [Op.in]: [...targetGender, ...universal]
      };
    }

    // Age Filter
    if (age) {
      const ageNum = parseInt(age);
      const orConditions = [];

      // 1. Keyword matching based on age
      if (ageNum < 18) orConditions.push({ [Op.like]: '%CHILD%' });
      if (ageNum >= 18 && ageNum < 65) orConditions.push({ [Op.like]: '%ADULT%' });
      if (ageNum >= 65) orConditions.push({ [Op.like]: '%OLDER_ADULT%' });

      // 2. Numeric logic
      orConditions.push({ [Op.regexp]: '[0-9]' }); 

      whereClause.age = {
         [Op.or]: orConditions
      };
    }

    const conditionText = String(condition || '').trim();
    const { query: extractedQuery, keywords: extractedKeywords } = await extractQueryAndKeywords(conditionText);
    const queryText = String(extractedQuery || conditionText).trim();

    const keywords = Array.isArray(extractedKeywords) ? extractedKeywords : [];
    if (keywords.length > 0) {
      const orClauses = [];
      keywords.slice(0, 12).forEach((kw) => {
        orClauses.push({ conditions: { [Op.like]: `%${kw}%` } });
        orClauses.push({ study_title: { [Op.like]: `%${kw}%` } });
      });
      whereClause[Op.or] = orClauses;
    } else if (queryText) {
      const safe = queryText.length > 120 ? queryText.slice(0, 120) : queryText;
      whereClause[Op.or] = [
        { conditions: { [Op.like]: `%${safe}%` } },
        { study_title: { [Op.like]: `%${safe}%` } },
      ];
    }

    const candidatePool = await registryConfig.model.findAll({
      where: whereClause,
      attributes: [registryConfig.idField, 'study_title', 'study_url', 'conditions', 'sex', 'age', 'phases', 'study_status'],
      limit: 5000,
    });

    if (candidatePool.length === 0) {
      ctx.body = { code: 200, data: [] };
      return;
    }

    // 3. BM25 Ranking
    const engine = bm25();
    engine.defineConfig({
      fldWeights: { title: 1, conditions: 2 },
      bm25Params: { k1: 1.2, b: 0.75 },
    });
    
    engine.definePrepTasks([
      winkNlpUtils.string.lowerCase,
      winkNlpUtils.string.removeExtraSpaces,
      winkNlpUtils.string.tokenize0,
    ]);

    const idMap = new Map();
    
    candidatePool.forEach((doc) => {
      const id = doc[registryConfig.idField];
      idMap.set(id, doc);
      engine.addDoc({
        title: doc.study_title || '',
        conditions: doc.conditions || '',
      }, id);
    });
    
    engine.consolidate();
    
    // Search
    // Note: condition string is the query
    const searchResults = queryText ? engine.search(queryText, 100) : [];
    

    let top100Trials = [];
    if (searchResults.length > 0) {
      top100Trials = searchResults
        .map((res) => idMap.get(res[0]))
        .filter((item) => item !== undefined);
    } else {
      top100Trials = candidatePool.slice(0, 100);
    }


    // 4. Fine Ranking (LLM) with top 100
    // Prepare data for LLM
    const trialsListStr = top100Trials.map(t => 
      `ID: ${t[registryConfig.idField]}, Title: ${t.study_title}, Cond: ${t.conditions}, Status: ${t.study_status}, Phase: ${t.phases}, Sex: ${t.sex}, Age: ${t.age}`
    ).join('\n');

    const prompt = `
    Patient Profile:
    - Age: ${age}
    - Gender: ${gender}
    - Condition: ${condition}
    - Registry: ${registryConfig.registry}

    Here are ${top100Trials.length} candidate clinical trials:
    ${trialsListStr}

    Task:
    1. STRICTLY Check if the patient meets the Age and Sex requirements of the trial.
       - Note on Age: The 'Age' field in trials can be keywords (CHILD, ADULT, OLDER_ADULT) or ranges (e.g., "18 Years to 65 Years").
       - Note on Sex: 'Both'/'All' means both genders are accepted.
    2. Analyze the suitability of the trial for the patient's condition.
    3. Select the top 10 most suitable trials.
    4. Assign a score (0-100) to each.
    5. Provide a brief reason for the match.

    Return the result STRICTLY in the following JSON format (no markdown, no extra text):
    {
      "trials": [
        { "trial_id": "...", "score": 95, "reason": "..." }
      ]
    }
    `;

    const completion = await aiClient.chat.completions.create({
      model: ALI_LLM_CONFIG.MODEL,
      messages: [
        { role: "system", content: "You are a clinical trial matching assistant. Output only valid JSON." },
        { role: "user", content: prompt }
      ],
      response_format: { type: "json_object" } 
    });

    let aiResultStr = completion.choices[0].message.content;
    
    // Clean up markdown code blocks if present
    aiResultStr = aiResultStr.replace(/```json/g, '').replace(/```/g, '');
    
    let rankedResults;
    try {
      rankedResults = JSON.parse(aiResultStr);
    } catch (e) {
      console.error("JSON Parse Error:", e);
      ctx.status = 500;
      ctx.body = { code: 500, msg: "AI response parsing failed" };
      return;
    }

    // Merge AI details back with full DB record (optional, or just return AI result merged with basic info)
    const trials = Array.isArray(rankedResults) ? rankedResults : rankedResults?.trials;
    const safeTrials = Array.isArray(trials) ? trials : [];

    const finalResults = safeTrials.map(r => {
      const original = top100Trials.find(t => String(t[registryConfig.idField]) === String(r.trial_id));
      const url =
        registryConfig.registry === 'usa'
          ? `https://clinicaltrials.gov/study/${r.trial_id}`
          : (original?.[registryConfig.urlField] || '');

      return {
        registry: registryConfig.registry,
        trial_id: r.trial_id,
        score: r.score,
        reason: r.reason,
        title: original ? original.study_title : '',
        url,
        conditions: original ? original.conditions : '',
        phases: original ? original.phases : '',
        study_status: original ? original.study_status : '',
      };
    });

    ctx.body = {
      code: 200,
      data: finalResults
    };

  } catch (error) {
    console.error("Match Patient Error:", error);
    ctx.status = 500;
    ctx.body = { code: 500, msg: error.message };
  }
}

async function matchTrial(ctx) {
  const startTime = Date.now();
  const perf = {}; // 性能统计对象
  console.log('>>> [MatchTrial] Start processing request...');

  try {
    const { registry, trialId, trialText, isRecruiting } = ctx.request.body;
    const registryConfig = getRegistryConfig(registry);

    if (!ALI_LLM_CONFIG.API_KEY) {
      ctx.status = 500;
      ctx.body = { code: 500, msg: 'Missing LLM API key. Set ALI_LLM_API_KEY or DASHSCOPE_API_KEY.' };
      return;
    }

    const idField = registryConfig.idField;
    let source = null;
    if (trialId) {
      source = await registryConfig.model.findByPk(String(trialId), {
        attributes: [idField, 'study_title', 'study_url', 'brief_summary', 'conditions', 'phases', 'study_status'],
      });
    }

    const sourceText = String(trialText || '').trim();
    if (!source && !sourceText) {
      ctx.status = 400;
      ctx.body = { code: 400, msg: 'trialId or trialText is required' };
      return;
    }

    const sourcePayload = source
      ? {
          trial_id: source[idField],
          title: source.study_title || '',
          conditions: source.conditions || '',
          phases: source.phases || '',
          status: source.study_status || '',
          summary: source.brief_summary || '',
        }
      : {
          trial_id: '',
          title: '',
          conditions: '',
          phases: '',
          status: '',
          summary: sourceText,
        };

    let keywords = [];
    let queryText = String(sourcePayload.title || sourcePayload.conditions || sourcePayload.summary || sourceText || '').trim();

    // === 阶段1：关键词提取 (LLM) ===
    const step1Start = Date.now();
    if (ALI_LLM_CONFIG.API_KEY) {
      // 有 Key 才调用 LLM 提取关键词
      const keywordPrompt = `
      You will receive one source clinical trial. Extract search keywords for retrieving similar trials.
      Output STRICT JSON in this format:
      {
        "query": "a short query string",
        "keywords": ["keyword1", "keyword2", "..."]
      }
  
      Source trial:
      ${JSON.stringify(sourcePayload)}
      `;
  
      try {
        const keywordCompletion = await aiClient.chat.completions.create({
          model: ALI_LLM_CONFIG.MODEL,
          messages: [
            { role: 'system', content: 'You extract search keywords. Output only valid JSON.' },
            { role: 'user', content: keywordPrompt },
          ],
          response_format: { type: 'json_object' },
        });
    
        let keywordResult = {};
        try {
          keywordResult = JSON.parse(String(keywordCompletion.choices?.[0]?.message?.content || '').replace(/```json/g, '').replace(/```/g, ''));
        } catch (e) {
          keywordResult = {};
        }
    
        keywords = Array.isArray(keywordResult.keywords)
          ? keywordResult.keywords.map((k) => String(k).trim()).filter((k) => k.length > 0)
          : [];
        
        if (keywordResult.query) {
           queryText = keywordResult.query;
        }
      } catch (e) {
        console.error("LLM keyword extraction failed:", e);
      }
    } else {
      // 无 Key：简单的关键词提取（分割标题）
      keywords = (sourcePayload.title || '').split(/\s+/).filter(w => w.length > 3).slice(0, 5);
    }
    perf.keywordExtraction = Date.now() - step1Start;
    console.log(`[MatchTrial] Step 1 (Keyword Extraction): ${perf.keywordExtraction}ms`);

    const whereClause = {};
    if (isRecruiting) {
      whereClause.study_status = {
        [Op.in]: [
          'RECRUITING',
          'Recruiting',
          'NOT_YET_RECRUITING',
          'Not yet recruiting',
          'ENROLLING_BY_INVITATION',
          'Enrolling by invitation',
        ],
      };
    }

    if (keywords.length > 0) {
      const orClauses = [];
      keywords.slice(0, 12).forEach((kw) => {
        orClauses.push({ study_title: { [Op.like]: `%${kw}%` } });
        orClauses.push({ conditions: { [Op.like]: `%${kw}%` } });
        orClauses.push({ brief_summary: { [Op.like]: `%${kw}%` } });
      });
      whereClause[Op.or] = orClauses;
    } else if (queryText) {
      whereClause[Op.or] = [
        { study_title: { [Op.like]: `%${queryText}%` } },
        { conditions: { [Op.like]: `%${queryText}%` } },
        { brief_summary: { [Op.like]: `%${queryText}%` } },
      ];
    }

    // === 阶段2：数据库检索 (DB Filter) ===
    const step2Start = Date.now();
    const candidatePool = await registryConfig.model.findAll({
      where: whereClause,
      attributes: [idField, 'study_title', 'study_url', 'brief_summary', 'conditions', 'phases', 'study_status', 'sex', 'age'],
      limit: 8000,
    });
    perf.dbQuery = Date.now() - step2Start;
    console.log(`[MatchTrial] Step 2 (DB Query): ${perf.dbQuery}ms. Candidates found: ${candidatePool.length}`);


    if (candidatePool.length === 0) {
      ctx.body = { code: 200, data: [] };
      return;
    }

    // === 阶段3：BM25 排序 (Ranking) ===
    const step3Start = Date.now();
    const engine = bm25();
    engine.defineConfig({
      fldWeights: { title: 1, conditions: 2, summary: 1 },
      bm25Params: { k1: 1.2, b: 0.75 },
    });
    engine.definePrepTasks([
      winkNlpUtils.string.lowerCase,
      winkNlpUtils.string.removeExtraSpaces,
      winkNlpUtils.string.tokenize0,
    ]);

    const idMap = new Map();
    candidatePool.forEach((doc) => {
      const id = doc[idField];
      idMap.set(String(id), doc);
      engine.addDoc(
        {
          title: doc.study_title || '',
          conditions: doc.conditions || '',
          summary: doc.brief_summary || '',
        },
        String(id)
      );
    });
    engine.consolidate();

    const bm25Results = queryText ? engine.search(queryText, 100) : [];
    const top100Trials =
      bm25Results.length > 0
        ? bm25Results.map((r) => idMap.get(String(r[0]))).filter((x) => x !== undefined).slice(0, 100)
        : candidatePool.slice(0, 100);
    perf.bm25Ranking = Date.now() - step3Start;
    console.log(`[MatchTrial] Step 3 (BM25 Ranking): ${perf.bm25Ranking}ms`);

    const candidatesText = top100Trials
      .map((t) => {
        const id = t[idField];
        const summary = String(t.brief_summary || '').slice(0, 300);
        return `ID: ${id}, Title: ${t.study_title || ''}, Cond: ${t.conditions || ''}, Phase: ${t.phases || ''}, Status: ${t.study_status || ''}, Summary: ${summary}`;
      })
      .join('\n');

    if (!ALI_LLM_CONFIG.API_KEY) {
      // 降级：直接返回 Top 10 BM25 结果
      const top10 = top100Trials.slice(0, 10).map((t, idx) => {
         const url = registryConfig.registry === 'usa'
            ? `https://clinicaltrials.gov/study/${t[idField]}`
            : (t.study_url || '');
         return {
            registry: registryConfig.registry,
            trial_id: t[idField],
            score: Math.max(100 - idx * 5, 60),
            reason: "Matched by BM25 keyword similarity (No LLM).",
            title: t.study_title,
            url,
            conditions: t.conditions,
            phases: t.phases,
            study_status: t.study_status,
         };
      });
      perf.totalTime = Date.now() - startTime;
      console.log(`[MatchTrial] Total Time (Fallback): ${perf.totalTime}ms. Performance:`, perf);
      ctx.body = { code: 200, data: top10 };
      return;
    }

    // === 阶段4：LLM 精排 (Fine Ranking) ===
    const step4Start = Date.now();
    const rankPrompt = `
    Source trial:
    ${JSON.stringify(sourcePayload)}

    Candidate trials (top 100 by BM25):
    ${candidatesText}

    Task:
    1. Choose the 10 most similar/matching trials to the source trial.
    2. Score each 0-100.
    3. Provide a concise reason.

    Output STRICT JSON:
    { "trials": [ { "trial_id": "...", "score": 90, "reason": "..." } ] }
    `;

    const rankCompletion = await aiClient.chat.completions.create({
      model: ALI_LLM_CONFIG.MODEL,
      messages: [
        { role: 'system', content: 'You are a clinical trial matching assistant. Output only valid JSON.' },
        { role: 'user', content: rankPrompt },
      ],
      response_format: { type: 'json_object' },
    });

    let rankedResults = {};
    try {
      rankedResults = JSON.parse(String(rankCompletion.choices?.[0]?.message?.content || '').replace(/```json/g, '').replace(/```/g, ''));
    } catch (e) {
      rankedResults = {};
    }
    perf.llmRanking = Date.now() - step4Start;
    console.log(`[MatchTrial] Step 4 (LLM Ranking): ${perf.llmRanking}ms`);

    const trials = Array.isArray(rankedResults) ? rankedResults : rankedResults?.trials;
    const safeTrials = Array.isArray(trials) ? trials : [];

    const finalResults = safeTrials.map((r) => {
      const original = top100Trials.find((t) => String(t[idField]) === String(r.trial_id));
      const url =
        registryConfig.registry === 'usa'
          ? `https://clinicaltrials.gov/study/${r.trial_id}`
          : (original?.study_url || '');
      return {
        registry: registryConfig.registry,
        trial_id: r.trial_id,
        score: r.score,
        reason: r.reason,
        title: original ? original.study_title : '',
        url,
        conditions: original ? original.conditions : '',
        phases: original ? original.phases : '',
        study_status: original ? original.study_status : '',
      };
    });

    perf.totalTime = Date.now() - startTime;
    console.log(`>>> [MatchTrial] Completed in ${perf.totalTime}ms. Detailed Stats:`, perf);

    ctx.body = { code: 200, data: finalResults };
  } catch (error) {
    console.error('Match Trial Error:', error);
    ctx.status = 500;
    ctx.body = { code: 500, msg: error.message };
  }
}

module.exports = {
  matchPatient,
  matchTrial,
};
