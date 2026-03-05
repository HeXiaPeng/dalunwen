import pandas as pd
import numpy as np
from pyecharts.charts import Line, Bar, Map, Pie
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import pycountry
from collections import Counter
from lifelines import CoxPHFitter

def generate_trend_plot(df: pd.DataFrame) -> str:
    """
    生成历年注册试验数量趋势分析图表 (Bar + Line)
    """
    data_cleaned = df.copy()
    
    # 筛选治疗方式
    if '治疗方式：大模型' in data_cleaned.columns:
        data_cleaned = data_cleaned[data_cleaned['治疗方式：大模型'].isin(['单纯双免治疗', '免疫联合局部治疗', '免疫联合靶向治疗'])]
    
    # 替换 Funder Type
    if 'Funder Type' in data_cleaned.columns:
        data_cleaned['Funder Type'] = data_cleaned['Funder Type'].replace({
            'FED': 'OTHER',
            'NETWORK': 'OTHER',
            'OTHER_GOV': 'OTHER'
        })

    if 'First Posted' not in data_cleaned.columns:
        # 如果没有 First Posted，尝试查找类似的列或返回空图
        return "<div>Error: Missing 'First Posted' column</div>"
    
    data_cleaned['First Posted'] = pd.to_datetime(data_cleaned['First Posted'])
    data_cleaned['Year'] = data_cleaned['First Posted'].dt.to_period('Y')
    registration_counts = data_cleaned.groupby('Year').size()
    
    # 处理 Phases
    if 'Phases' in data_cleaned.columns:
        data_cleaned.fillna({'Phases': '未指定'}, inplace=True)
        phase_counts = data_cleaned.groupby(['Year', 'Phases']).size().unstack(fill_value=0)
        
        desired_order = ['EARLY_PHASE1', 'PHASE1', 'PHASE1|PHASE2', 'PHASE2', 'PHASE2|PHASE3', 'PHASE3', 'PHASE4']
        existing_order = [col for col in desired_order if col in phase_counts.columns]
        other_cols = [col for col in phase_counts.columns if col not in desired_order]
        final_order = existing_order + other_cols
        phase_counts = phase_counts[final_order]
        
        rename_map = {
            'EARLY_PHASE1': 'Early Phase1',
            'PHASE1': 'Phase1',
            'PHASE1|PHASE2': 'Phase1 | Phase2',
            'PHASE2': 'Phase2',
            'PHASE2|PHASE3': 'Phase2 | Phase3',
            'PHASE3':'Phase3',
            'PHASE4':'Phase4',
            'Not reported': 'Not reported'
        }
        phase_counts.rename(columns=rename_map, inplace=True)
    else:
        phase_counts = pd.DataFrame()

    y_data = registration_counts.fillna(0).astype(int).values
    x_data = [str(year) for year in registration_counts.index]

    if len(x_data) > 1:
        x_numeric = np.arange(len(x_data))
        trend_coefficients = np.polyfit(x_numeric, y_data, deg=1)
        trend_line = np.polyval(trend_coefficients, x_numeric)
    else:
        trend_line = y_data

    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE, width="100%", height="500px"))
        .add_xaxis(x_data)
    )

    for phase in phase_counts.columns:
        bar.add_yaxis(
            f"{phase}", phase_counts[phase].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            stack="stack1"
        )

    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="历年注册试验数量趋势分析"),
        xaxis_opts=opts.AxisOpts(
            name="Year of first public information",
            name_location="center",
            name_gap=30,
            axislabel_opts=opts.LabelOpts(rotate=0, interval=0),
        ),
        yaxis_opts=opts.AxisOpts(
            name="Number of Trials", 
            name_location="center",
            name_gap=40,
            min_=0,
        ),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
        legend_opts=opts.LegendOpts(pos_left="right", orient="vertical", pos_top="15%")
    )

    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis("Total Trials", y_data.tolist(), is_smooth=True, symbol="circle", label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("Trendline", trend_line.tolist(), linestyle_opts=opts.LineStyleOpts(type_="dashed", color="black"), symbol='arrow', symbol_size=10, label_opts=opts.LabelOpts(is_show=False), itemstyle_opts=opts.ItemStyleOpts(color="rgba(0, 0, 0, 0)"))
    )

    bar.overlap(line)
    return bar.render_embed()

def generate_treatment_plot(df: pd.DataFrame) -> str:
    """
    Annual Registration Counts for Different Treatment Approaches
    """
    data_cleaned = df.copy()
    treatment_col = '治疗方式：大模型'
    
    if treatment_col not in data_cleaned.columns or 'First Posted' not in data_cleaned.columns:
         return "<div>Error: Missing required columns for treatment plot</div>"

    data_cleaned['First Posted'] = pd.to_datetime(data_cleaned['First Posted'])
    data_cleaned['Year'] = data_cleaned['First Posted'].dt.to_period('Y')
    registration_counts = data_cleaned.groupby('Year').size()
    
    data_cleaned.fillna({treatment_col: '未指定'}, inplace=True)
    
    confirmed_counts = data_cleaned.groupby(['Year', treatment_col]).size().unstack(fill_value=0)
    confirmed_counts.rename(columns={
        "免疫联合局部治疗": "Immunotherapy combined with local therapy",
        "免疫联合靶向治疗": "Immunotherapy combined with targeted therapy",
        "单纯双免治疗": "Dual immunotherapy",
    }, inplace=True)
    
    y_data = registration_counts.fillna(0).astype(int).values
    x_data = [str(year) for year in registration_counts.index]
    
    bar = Bar(init_opts=opts.InitOpts(width="100%", height="560px"))
    bar.add_xaxis(x_data)
    
    for confirmed in confirmed_counts.columns:
        bar.add_yaxis(
            f"{confirmed}", confirmed_counts[confirmed].tolist(),
            label_opts=opts.LabelOpts(is_show=False),
            stack="stack1"
        )
        
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="不同治疗方式的年度注册数量"),
        xaxis_opts=opts.AxisOpts(name="Year of first public information", name_location="center", name_gap=30),
        yaxis_opts=opts.AxisOpts(name="Number of Trials", name_location="center", name_gap=30),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(
            type_="scroll",
            pos_right="0%",
            pos_top="12%",
            orient="vertical",
            item_width=12,
            item_height=8,
            textstyle_opts=opts.TextStyleOpts(font_size=11)
        )
    )
    
    line = Line().add_xaxis(x_data).add_yaxis("Total trials", y_data.tolist(), is_smooth=True, symbol="circle")
    bar.overlap(line)
    
    return bar.render_embed()

def standardize_country_name(country_name):
    try:
        return pycountry.countries.lookup(country_name).name
    except LookupError:
        return country_name

def generate_map_plot(df: pd.DataFrame) -> str:
    """
    Clinical Trials Count by Country
    """
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
    
    if 'Locations' not in df.columns:
        return "<div>Error: Missing 'Locations' column</div>"

    locations = df['Locations'].dropna().tolist()
    country_counts = Counter()
    for location in locations:
        unique_contry = set()
        for place in str(location).split('|'):
            country = place.split(',')[-1].strip()
            if country == 'Republic of':
                parts = place.split(',')
                if len(parts) >= 2:
                    country = parts[-1].strip() + ' ' + parts[-2].strip()
            country = replacement_dict.get(country, country)
            country = replacement_dict2.get(country, country)
            if country not in unique_contry:
                unique_contry.add(country)
        for country in unique_contry:
            country_counts[country] += 1
            
    country_df = pd.DataFrame(country_counts.items(), columns=['Country', 'Clinical Trials Count'])
    country_df = country_df.sort_values(by='Clinical Trials Count', ascending=False).reset_index(drop=True)
    country_df['Country'] = country_df['Country'].apply(standardize_country_name)
    country_df['Country'] = country_df['Country'].replace({'Korea, Republic of': 'South Korea', "Côte d'Ivoire": 'Ivory Coast'})
    country_df = country_df.groupby("Country", as_index=False).sum()
    
    map_data = [list(z) for z in zip(country_df['Country'], country_df['Clinical Trials Count'])]
    
    c = (
        Map(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add("", map_data, "world")
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Clinical Trials Count by Country"),
            visualmap_opts=opts.VisualMapOpts(
                max_=max(country_df['Clinical Trials Count']) if not country_df.empty else 100,
                pos_left="10%",
                pos_top="57%"
            ),
        )
    )
    return c.render_embed()

def generate_weights_plot(df: pd.DataFrame) -> str:
    """
    Weights of Main Categories (Cox Model)
    """
    if 'Study Status' not in df.columns:
         return "<div>Error: Missing 'Study Status' column</div>"
         
    filtered_df = df[df['Study Status'].isin(['COMPLETED', 'WITHDRAWN', 'TERMINATED', 'SUSPENDED'])].copy()
    if filtered_df.empty:
        return "<div>Error: No valid data for survival analysis</div>"

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
        "单纯双免治疗": "Dual Immune Therapy",
    }
    if 'Treatment' in filtered_df.columns:
        filtered_df['Treatment'] = filtered_df['Treatment'].replace(translation_dict)
    
    cols_to_use = ['status', 'Duration', 'Phases', 'Treatment', 'Sample Size Category', 'Funder Type', 'Allocation', 'Intervention Model', 'Primary Purpose', 'Masking']
    available_cols = [c for c in cols_to_use if c in filtered_df.columns]
    
    filtered_df = filtered_df[available_cols]
    
    # Dummy variables
    cat_cols = [c for c in available_cols if c not in ['status', 'Duration']]
    filtered_df = pd.get_dummies(filtered_df, columns=cat_cols, drop_first=True)
    
    filtered_df['Duration'] = pd.to_numeric(filtered_df['Duration'], errors='coerce')
    filtered_df.dropna(subset=['Duration', 'status'], inplace=True)
    
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(filtered_df, duration_col="Duration", event_col="status")
        
        summary_df = cph.summary.rename_axis('covariate').reset_index()
        summary_df['log_HR'] = np.log(summary_df['exp(coef)'])
        summary_df['weight'] = summary_df['log_HR'].abs()
        summary_df = summary_df[summary_df['weight'] > 0]
        
        main_categories = ['Phases', 'Treatment', 'Sample Size Category', 'Funder Type', 'Allocation', 'Intervention Model', 'Primary Purpose', 'Masking']
        
        def map_to_main_category(var):
            for category in main_categories:
                if var.startswith(category + '_'):
                    return category
            return None
            
        summary_df['main_category'] = summary_df['covariate'].apply(map_to_main_category)
        filtered_summary = summary_df[summary_df['main_category'].notnull()]
        
        aggregated_weights = filtered_summary.groupby('main_category')['weight'].sum().reset_index()
        top_vars = aggregated_weights
        
        pie = (
            Pie(init_opts=opts.InitOpts(width="100%", height="500px"))
            .add(
                "",
                [list(z) for z in zip(top_vars['main_category'], top_vars['weight'])],
                radius=["40%", "75%"],
                rosetype="radius",
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Weights of Main Categories"),
                legend_opts=opts.LegendOpts(is_show=True, orient="vertical", pos_top="middle", pos_left="10%")
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
        )
        return pie.render_embed()
        
    except Exception as e:
        return f"<div>Error in Cox Model: {str(e)}</div>"

def generate_failure_plot(df: pd.DataFrame) -> str:
    """
    Failure Counts (Pie Chart)
    """
    if '失败分类' not in df.columns:
        return "<div>Error: Missing '失败分类' column</div>"
        
    failure_counts = df['失败分类'].value_counts()
    failure_counts = failure_counts.rename(index={"No reason provided": "not report"})
    
    failure_counts_processed = failure_counts[failure_counts > 1]
    failure_counts_processed["else"] = failure_counts[failure_counts <= 1].sum()
    
    data_pairs = [[f.title() if f != 'PI' else f, int(failure_counts_processed[f])] for f in failure_counts_processed.index]
    
    c = (
        Pie(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add("", data_pairs)
        .set_global_opts(title_opts=opts.TitleOpts(title="Failure Reasons Distribution"))
        .set_series_opts(
            label_opts=opts.LabelOpts(formatter="{b}: {c}({d}%)", font_size=15),
            legend_opts=opts.LegendOpts(is_show=True, orient="vertical", pos_right="right", pos_top="middle")
        )
    )
    return c.render_embed()

def generate_survival_plots(df: pd.DataFrame) -> str:
    """
    Generate Survival Curves for categorical variables (Line Chart)
    """
    if 'Study Status' not in df.columns:
         return "<div>Error: Missing 'Study Status' column</div>"
         
    filtered_df = df[df['Study Status'].isin(['COMPLETED', 'WITHDRAWN', 'TERMINATED', 'SUSPENDED'])].copy()
    if filtered_df.empty:
        return "<div>Error: No valid data for survival analysis</div>"

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
        "单纯双免治疗": "Dual Immune Therapy",
    }
    if 'Treatment' in filtered_df.columns:
        filtered_df['Treatment'] = filtered_df['Treatment'].replace(translation_dict)
    
    categorical_vars = [
        'Phases',
        'Treatment',
        'Sample Size Category',
        'Funder Type',
        'Allocation',
        'Intervention Model',
        'Primary Purpose',
        'Masking'
    ]
    
    cols_to_use = ['status', 'Duration'] + [c for c in categorical_vars if c in filtered_df.columns]
    filtered_df = filtered_df[cols_to_use]
    
    # Dummy variables
    cat_cols = [c for c in cols_to_use if c not in ['status', 'Duration']]
    filtered_df = pd.get_dummies(filtered_df, columns=cat_cols, drop_first=True)
    all_columns = filtered_df.columns
    
    filtered_df['Duration'] = pd.to_numeric(filtered_df['Duration'], errors='coerce')
    filtered_df.dropna(subset=['Duration', 'status'], inplace=True)
    
    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(filtered_df, duration_col="Duration", event_col="status")
        
        # We will create a single chart with multiple tabs or just combine them?
        # Pyecharts doesn't support subplots natively in the same way as matplotlib.
        # We can use Tab or Grid, but for now let's return one combined chart for the first variable or all in a Tab.
        from pyecharts.charts import Tab
        
        tab = Tab()
        
        mean_covariates = filtered_df.mean().copy()
        
        for cat_var in categorical_vars:
            if cat_var not in cols_to_use: # check if original column existed
                continue
                
            line = Line(init_opts=opts.InitOpts(width="100%", height="500px"))
            cat_dummy_cols = [col for col in all_columns if col.startswith(cat_var + "_")]
            
            # Total Survival Curve
            survival_function = cph.predict_survival_function(mean_covariates.to_frame().T)
            line.add_xaxis([str(i) for i in survival_function.index.tolist()])
            line.add_yaxis("Total Survival Curve", survival_function.values.flatten().tolist(), 
                           is_symbol_show=False, is_smooth=True, 
                           linestyle_opts=opts.LineStyleOpts(width=3, color="black"))
            
            # Baseline
            baseline_covariates = mean_covariates.copy()
            for col in cat_dummy_cols:
                baseline_covariates[col] = 0
            sf_baseline = cph.predict_survival_function(baseline_covariates.to_frame().T)
            line.add_yaxis("Baseline Category", sf_baseline.values.flatten().tolist(), is_symbol_show=False, is_step=True)
            
            # Other categories
            for dummy_col in cat_dummy_cols:
                scenario_covariates = mean_covariates.copy()
                for col in cat_dummy_cols:
                    scenario_covariates[col] = 0
                scenario_covariates[dummy_col] = 1
                
                sf = cph.predict_survival_function(scenario_covariates.to_frame().T)
                category_name = dummy_col.replace(cat_var + "_", "").replace("_", " ")
                
                line.add_yaxis(category_name, sf.values.flatten().tolist(), is_symbol_show=False, is_step=True)
                
            line.set_global_opts(
                title_opts=opts.TitleOpts(title=f"Survival by {cat_var}"),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(name="Days"),
                yaxis_opts=opts.AxisOpts(name="Survival Probability"),
                legend_opts=opts.LegendOpts(type_="scroll", pos_right="0%", orient="vertical")
            )
            tab.add(line, cat_var)
            
        return tab.render_embed()
        
    except Exception as e:
        return f"<div>Error in Cox Model Survival Plots: {str(e)}</div>"
