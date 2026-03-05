from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import os
import numpy as np
from collections import Counter
from lifelines import CoxPHFitter
from plots import (
    generate_trend_plot, 
    generate_treatment_plot, 
    generate_map_plot, 
    generate_weights_plot, 
    generate_failure_plot,
    generate_survival_plots
)

router = APIRouter(
    prefix="/api/analysis",
    tags=["analysis"]
)

# ==========================================
# Helper
# ==========================================
def make_html_response(chart_html: str) -> str:
    """
    Wrap the chart HTML snippet in a full HTML document with ECharts library.
    """
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Chart</title>
        <script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            .chart-container {{ width: 100%; height: 100vh; }}
        </style>
    </head>
    <body>
        {chart_html}
    </body>
    </html>
    """

# ==========================================
# Data Loading
# ==========================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "../mock/11月19日需要确认excel_飞书.xlsx")

def get_data(sheet_name: str = '免疫联合治疗'):
    """
    Load data from the mock Excel file.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    
    try:
        df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
        return df
    except Exception as e:
        raise Exception(f"Failed to load data from {DATA_PATH}: {str(e)}")

def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 1)

def build_trend_insight(df: pd.DataFrame):
    data_cleaned = df.copy()
    if '治疗方式：大模型' in data_cleaned.columns:
        data_cleaned = data_cleaned[data_cleaned['治疗方式：大模型'].isin(['单纯双免治疗', '免疫联合局部治疗', '免疫联合靶向治疗'])]
    if 'First Posted' not in data_cleaned.columns:
        return {"summary": "缺少时间字段，无法生成趋势分析。", "bullets": ["请检查 First Posted 字段。"]}
    data_cleaned['First Posted'] = pd.to_datetime(data_cleaned['First Posted'], errors='coerce')
    data_cleaned = data_cleaned[data_cleaned['First Posted'].notna()].copy()
    if data_cleaned.empty:
        return {"summary": "有效时间数据不足，无法生成趋势分析。", "bullets": ["请补充试验发布时间。"]}
    data_cleaned['Year'] = data_cleaned['First Posted'].dt.year
    yearly = data_cleaned.groupby('Year').size().sort_index()
    start_year = int(yearly.index.min())
    end_year = int(yearly.index.max())
    start_count = int(yearly.iloc[0])
    end_count = int(yearly.iloc[-1])
    growth = _safe_pct(end_count - start_count, start_count if start_count else 1)
    peak_year = int(yearly.idxmax())
    peak_count = int(yearly.max())
    recent_avg = round(float(yearly.tail(min(3, len(yearly))).mean()), 1)
    return {
        "summary": f"{start_year}-{end_year} 年注册数量整体上升，峰值出现在 {peak_year} 年。",
        "bullets": [
            f"起始年份 {start_year} 年为 {start_count} 项，最新 {end_year} 年为 {end_count} 项，累计增幅约 {growth}%。",
            f"历史峰值为 {peak_count} 项，出现在 {peak_year} 年。",
            f"近 {min(3, len(yearly))} 年年均注册量约 {recent_avg} 项。"
        ]
    }

def build_treatment_insight(df: pd.DataFrame):
    treatment_col = '治疗方式：大模型'
    if treatment_col not in df.columns:
        return {"summary": "缺少治疗方式字段，无法生成治疗方式分析。", "bullets": ["请检查治疗方式：大模型字段。"]}
    counts = df[treatment_col].fillna('未指定').value_counts()
    total = int(counts.sum())
    top_name = str(counts.index[0]) if not counts.empty else '未指定'
    top_count = int(counts.iloc[0]) if not counts.empty else 0
    top_share = _safe_pct(top_count, total if total else 1)
    if 'First Posted' in df.columns:
        data = df.copy()
        data['First Posted'] = pd.to_datetime(data['First Posted'], errors='coerce')
        data = data[data['First Posted'].notna()].copy()
        data['Year'] = data['First Posted'].dt.year
        yearly_treatment = data.groupby(['Year', treatment_col]).size().unstack(fill_value=0)
        latest_year = int(yearly_treatment.index.max()) if not yearly_treatment.empty else None
        latest_leader = yearly_treatment.loc[latest_year].idxmax() if latest_year is not None else top_name
    else:
        latest_year = None
        latest_leader = top_name
    latest_desc = f"{latest_year} 年领先方案为 {latest_leader}。" if latest_year is not None else "无法识别最新年份领先方案。"
    return {
        "summary": f"治疗方式结构呈头部集中，{top_name} 当前占比最高。",
        "bullets": [
            f"{top_name} 共 {top_count} 项，占全部治疗方式样本的 {top_share}%。",
            f"共识别 {len(counts)} 类治疗方式，结构分布存在明显长尾。",
            latest_desc
        ]
    }

def build_map_insight(df: pd.DataFrame):
    if 'Locations' not in df.columns:
        return {"summary": "缺少 Locations 字段，无法生成分布分析。", "bullets": ["请检查 Locations 字段。"]}
    replacement_dict = {
        'Korea, Republic of': 'South Korea', 'Republic of Korea': 'South Korea',
        'Taiwan 10002': 'Taiwan', 'United Kingdom London': 'United Kingdom',
        'United Kingdom W12 0HS': 'United Kingdom', 'Taiwan 10048': 'Taiwan',
        'United Kingdom OX3 7LE': 'United Kingdom', 'Taiwan 100': 'Taiwan',
        'Taiwan 11217': 'Taiwan', 'Vietnam 70000': 'Vietnam',
        'Vietnam 700000': 'Vietnam', 'Turkey 01120': 'Turkey',
        'Robert H. Lurie Comprehensive Cancer Center': 'United States',
        'Taiwan 112201': 'Taiwan', 'United Kingdom SE5 9RS': 'United Kingdom',
        'United Kingdom NG5 1PB': 'United Kingdom', 'United Kingdom M2O 4BX': 'United Kingdom',
        'Taiwan 33305': 'Taiwan', 'Switzerland 8091': 'Switzerland',
        'Turkey 41380': 'Turkey', 'Spain 08035': 'Spain',
        'Taiwan Taipei': 'Taiwan', 'Turkey Multiple Locations': 'Turkey',
        'United Kingdom M20 4BX': 'United Kingdom', 'Singapore 169610': 'Singapore',
        'Hospital General Universitario Gregorio Marañón': 'Spain',
        'Spain 28007': 'Spain', 'Turkey 34010': 'Turkey',
        'United Kingdom NW3 2QG': 'United Kingdom', 'United Kingdom BT9 7AB': 'United Kingdom',
        'Taiwan 333': 'Taiwan', 'Vietnam Hochiminh': 'Vietnam',
        "Côte d'Ivoire": 'Ivory Coast'
    }
    replacement_dict2 = {'Taiwan': 'China', 'Hong Kong': 'China'}
    locations = df['Locations'].dropna().tolist()
    country_counts = Counter()
    for location in locations:
        unique_country = set()
        for place in str(location).split('|'):
            country = place.split(',')[-1].strip()
            if country == 'Republic of':
                parts = place.split(',')
                if len(parts) >= 2:
                    country = parts[-1].strip() + ' ' + parts[-2].strip()
            country = replacement_dict.get(country, country)
            country = replacement_dict2.get(country, country)
            if country not in unique_country:
                unique_country.add(country)
        for country in unique_country:
            country_counts[country] += 1
    if not country_counts:
        return {"summary": "暂无可用地区分布数据。", "bullets": ["请检查 Locations 字段内容。"]}
    total = sum(country_counts.values())
    top3 = Counter(country_counts).most_common(3)
    top_country, top_count = top3[0]
    top_share = _safe_pct(top_count, total)
    top_desc = "、".join([f"{name}({count})" for name, count in top3])
    return {
        "summary": f"全球试验分布呈集中态势，{top_country} 位居首位。",
        "bullets": [
            f"覆盖国家/地区共 {len(country_counts)} 个，Top1 为 {top_country}，数量 {top_count}。",
            f"Top1 占比约 {top_share}%，头部国家集聚效应明显。",
            f"Top3 分别为：{top_desc}。"
        ]
    }

def build_weights_insight(df: pd.DataFrame):
    if 'Study Status' not in df.columns:
        return {"summary": "缺少 Study Status 字段，无法生成权重分析。", "bullets": ["请检查 Study Status 字段。"]}
    filtered_df = df[df['Study Status'].isin(['COMPLETED', 'WITHDRAWN', 'TERMINATED', 'SUSPENDED'])].copy()
    if filtered_df.empty:
        return {"summary": "有效状态样本不足，无法生成权重分析。", "bullets": ["请补充 COMPLETED/WITHDRAWN/TERMINATED/SUSPENDED 数据。"]}
    filtered_df['status'] = (filtered_df['Study Status'] == 'COMPLETED').astype(int)
    filtered_df['Primary Completion Date'] = pd.to_datetime(filtered_df['Primary Completion Date'], errors='coerce')
    filtered_df['Start Date'] = pd.to_datetime(filtered_df['Start Date'], errors='coerce')
    filtered_df['Duration'] = (filtered_df['Primary Completion Date'] - filtered_df['Start Date']).dt.days
    filtered_df = filtered_df[filtered_df['Duration'] >= 0]
    if '治疗方式：大模型' in filtered_df.columns:
        filtered_df.rename(columns={'治疗方式：大模型': 'Treatment'}, inplace=True)
    translation_dict = {
        "免疫联合局部治疗": "Immune Combination Local Therapy",
        "免疫联合靶向治疗": "Immune Combination Targeted Therapy",
        "单纯双免治疗": "Dual Immune Therapy"
    }
    if 'Treatment' in filtered_df.columns:
        filtered_df['Treatment'] = filtered_df['Treatment'].replace(translation_dict)
    cols_to_use = ['status', 'Duration', 'Phases', 'Treatment', 'Sample Size Category', 'Funder Type', 'Allocation', 'Intervention Model', 'Primary Purpose', 'Masking']
    available_cols = [c for c in cols_to_use if c in filtered_df.columns]
    model_df = filtered_df[available_cols]
    cat_cols = [c for c in available_cols if c not in ['status', 'Duration']]
    model_df = pd.get_dummies(model_df, columns=cat_cols, drop_first=True)
    model_df['Duration'] = pd.to_numeric(model_df['Duration'], errors='coerce')
    model_df.dropna(subset=['Duration', 'status'], inplace=True)
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(model_df, duration_col="Duration", event_col="status")
        summary_df = cph.summary.rename_axis('covariate').reset_index()
        summary_df['log_HR'] = np.log(summary_df['exp(coef)'])
        summary_df['weight'] = summary_df['log_HR'].abs()
        summary_df = summary_df[summary_df['weight'] > 0]
        main_categories = ['Phases', 'Treatment', 'Sample Size Category', 'Funder Type', 'Allocation', 'Intervention Model', 'Primary Purpose', 'Masking']
        def map_to_main_category(var):
            for category in main_categories:
                if str(var).startswith(category + '_'):
                    return category
            return None
        summary_df['main_category'] = summary_df['covariate'].apply(map_to_main_category)
        filtered_summary = summary_df[summary_df['main_category'].notnull()]
        aggregated_weights = filtered_summary.groupby('main_category')['weight'].sum().reset_index().sort_values('weight', ascending=False)
        if aggregated_weights.empty:
            return {"summary": "当前样本不足以形成稳定权重分布。", "bullets": ["请补充更多已完成与终止样本。"]}
        top = aggregated_weights.iloc[0]
        second = aggregated_weights.iloc[1] if len(aggregated_weights) > 1 else None
        total_weight = float(aggregated_weights['weight'].sum())
        top_share = _safe_pct(float(top['weight']), total_weight if total_weight else 1.0)
        second_text = f"次要影响维度为 {second['main_category']}（占比约 {_safe_pct(float(second['weight']), total_weight if total_weight else 1.0)}%）。" if second is not None else "当前仅识别到一个显著影响维度。"
        return {
            "summary": f"主要分类变量中，{top['main_category']} 对结局区分的影响权重最高。",
            "bullets": [
                f"Top1 维度为 {top['main_category']}，权重占比约 {top_share}%。",
                second_text,
                f"当前有效权重维度共 {len(aggregated_weights)} 类。"
            ]
        }
    except Exception:
        return {"summary": "权重模型拟合未成功，建议检查样本完整性。", "bullets": ["可先检查 Duration 与 Study Status 字段缺失情况。"]}

def build_failure_insight(df: pd.DataFrame):
    if '失败分类' not in df.columns:
        return {"summary": "缺少失败分类字段，无法生成失败原因分析。", "bullets": ["请检查 失败分类 字段。"]}
    failure_counts = df['失败分类'].value_counts()
    if failure_counts.empty:
        return {"summary": "暂无失败原因数据。", "bullets": ["请补充失败样本后重试。"]}
    failure_counts = failure_counts.rename(index={"No reason provided": "not report"})
    processed = failure_counts[failure_counts > 1].copy()
    processed.loc['else'] = int(failure_counts[failure_counts <= 1].sum())
    total = int(processed.sum()) if int(processed.sum()) > 0 else 1
    top_reason = str(processed.idxmax())
    top_count = int(processed.max())
    top_share = _safe_pct(top_count, total)
    return {
        "summary": f"失败原因分布中，{top_reason} 是当前最主要风险来源。",
        "bullets": [
            f"{top_reason} 数量为 {top_count}，占比约 {top_share}%。",
            f"已识别主要失败类别 {len(processed)} 类（含 else 合并项）。",
            "建议优先针对 Top 原因优化方案设计与入排标准。"
        ]
    }

def build_survival_insight(df: pd.DataFrame):
    if 'Study Status' not in df.columns:
        return {"summary": "缺少 Study Status 字段，无法生成生存分析。", "bullets": ["请检查 Study Status 字段。"]}
    filtered_df = df[df['Study Status'].isin(['COMPLETED', 'WITHDRAWN', 'TERMINATED', 'SUSPENDED'])].copy()
    if filtered_df.empty:
        return {"summary": "有效状态样本不足，无法生成生存分析。", "bullets": ["请补充结局状态相关样本。"]}
    filtered_df['status'] = (filtered_df['Study Status'] == 'COMPLETED').astype(int)
    filtered_df['Primary Completion Date'] = pd.to_datetime(filtered_df['Primary Completion Date'], errors='coerce')
    filtered_df['Start Date'] = pd.to_datetime(filtered_df['Start Date'], errors='coerce')
    filtered_df['Duration'] = (filtered_df['Primary Completion Date'] - filtered_df['Start Date']).dt.days
    filtered_df = filtered_df[filtered_df['Duration'] >= 0]
    if filtered_df.empty:
        return {"summary": "可用于生存分析的时长样本不足。", "bullets": ["请检查 Start Date 与 Primary Completion Date 数据。"]}
    completion_rate = round(float(filtered_df['status'].mean() * 100), 1)
    median_duration = int(filtered_df['Duration'].median()) if filtered_df['Duration'].notna().any() else 0
    treatment_note = "治疗方式维度暂无可用分层信息。"
    if '治疗方式：大模型' in filtered_df.columns:
        by_treatment = filtered_df.groupby('治疗方式：大模型')['status'].mean().sort_values(ascending=False)
        if not by_treatment.empty:
            best = by_treatment.index[0]
            best_rate = round(float(by_treatment.iloc[0] * 100), 1)
            treatment_note = f"治疗方式分层中，{best} 完成率最高（约 {best_rate}%）。"
    return {
        "summary": "生存分析显示不同分类变量间存在明显结局差异。",
        "bullets": [
            f"当前可分析样本完成率约 {completion_rate}%。",
            f"样本中位持续时间约 {median_duration} 天。",
            treatment_note
        ]
    }

# ==========================================
# Analysis Endpoints
# ==========================================

@router.get("/trend")
async def analyze_trend():
    """
    [MOCK] 历年注册试验数量趋势分析图表 (Bar + Line)
    """
    try:
        df = get_data(sheet_name='免疫联合治疗')
        chart_html = generate_trend_plot(df)
        return HTMLResponse(content=make_html_response(chart_html))
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})

@router.get("/treatment")
async def analyze_treatment():
    """
    [MOCK] 不同治疗方式的年度注册数量 (Stacked Bar + Line)
    """
    try:
        df = get_data(sheet_name='免疫联合治疗')
        chart_html = generate_treatment_plot(df)
        return HTMLResponse(content=make_html_response(chart_html))
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})

@router.get("/map")
async def analyze_map():
    """
    [MOCK] Clinical Trials Count by Country (World Map)
    """
    try:
        df = get_data(sheet_name='免疫联合治疗')
        chart_html = generate_map_plot(df)
        return HTMLResponse(content=make_html_response(chart_html))
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})

@router.get("/weights")
async def analyze_weights():
    """
    [MOCK] Weights of Main Categories (Cox Model -> Pie Chart)
    """
    try:
        df = get_data(sheet_name='免疫联合治疗')
        chart_html = generate_weights_plot(df)
        return HTMLResponse(content=make_html_response(chart_html))
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})

@router.get("/failure")
async def analyze_failure():
    """
    [MOCK] Failure Reasons Distribution (Pie Chart)
    """
    try:
        # Note: Failure data is in a different sheet
        df = get_data(sheet_name='含有失败原因')
        chart_html = generate_failure_plot(df)
        return HTMLResponse(content=make_html_response(chart_html))
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})

@router.get("/survival")
async def analyze_survival():
    """
    [MOCK] Survival Curves for categorical variables (Tabs with Line Charts)
    """
    try:
        df = get_data(sheet_name='免疫联合治疗')
        chart_html = generate_survival_plots(df)
        return HTMLResponse(content=make_html_response(chart_html))
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})

@router.get("/insights")
async def analyze_insights(registry: str = "china", repository: str = "default", query: str = ""):
    try:
        df_main = get_data(sheet_name='免疫联合治疗')
        df_failure = get_data(sheet_name='含有失败原因')
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "context": {
                    "registry": registry,
                    "repository": repository,
                    "query": query
                },
                "trend": build_trend_insight(df_main),
                "treatment": build_treatment_insight(df_main),
                "map": build_map_insight(df_main),
                "weights": build_weights_insight(df_main),
                "failure": build_failure_insight(df_failure),
                "survival": build_survival_insight(df_main)
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"code": 500, "msg": f"Analysis failed: {str(e)}"})
