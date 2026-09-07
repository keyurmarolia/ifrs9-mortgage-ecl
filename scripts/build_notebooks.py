from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

SETUP = """from pathlib import Path
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path.cwd().resolve()
if not (ROOT / "config").exists():
    ROOT = ROOT.parent
DB = ROOT / "database" / "ifrs9_ecl.sqlite3"
CFG = yaml.safe_load((ROOT / "config" / "project.yaml").read_text())

def query(sql):
    with sqlite3.connect(DB) as connection:
        return pd.read_sql_query(sql, connection)

pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")
plt.rcParams["figure.figsize"] = (9, 4)
"""


def M(text):
    return nbf.v4.new_markdown_cell(text.strip())


def C(text):
    return nbf.v4.new_code_cell(text.strip())


def make(title, purpose, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [M("# " + title + "\n\n" + purpose), C(SETUP), *cells]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    }
    return nb


NOTEBOOKS = {
"00_project_flow.ipynb": make(
"Project flow",
"This notebook shows the calculation order and final portfolio totals.",
[
M("""Calculation order

1. Load loan and macro history.
2. Define default and build model features.
3. Estimate 12-month PD and monthly hazard.
4. Assign stages and build macro scenarios.
5. Create monthly PD, LGD, EAD and discount-factor terms.
6. Calculate scenario ECL and apply scenario weights.
7. Reconcile loan, stage and portfolio totals."""),
C('query("select * from executive_summary")'),
C('query("select * from stage_summary order by stage")'),
C('query("select * from scenario_summary order by scenario")'),
M("The detailed calculation is stored in ecl_projection_cube. Each row is one loan, one scenario and one future month.")
]),

"01_data_quality_and_eda.ipynb": make(
"Data quality and portfolio EDA",
"The source tables, observation periods and main risk characteristics are reviewed before modelling.",
[
C("""query('''
select 'origination_mortgages' table_name, count(*) rows, count(distinct loan_id) loans from origination_mortgages
union all
select 'monthly_loan_performance', count(*), count(distinct loan_id) from monthly_loan_performance
union all
select 'reporting_date_portfolio', count(*), count(distinct loan_id) from reporting_date_portfolio
''')"""),
C("""query('''
select min(period) first_month, max(period) last_month,
avg(current_actual_upb) average_upb, avg(current_dpd) average_dpd
from monthly_loan_performance
''')"""),
C("""query('''
select
sum(case when period is null then 1 else 0 end) missing_periods,
count(*)-count(distinct loan_id || '|' || period) duplicate_loan_months,
sum(case when current_actual_upb<0 then 1 else 0 end) negative_balances
from monthly_loan_performance
''')"""),
C("""portfolio = query('''select classic_fico, original_dti, current_ltv,
current_actual_upb, current_dpd, loan_age from reporting_date_portfolio''')
portfolio.describe(percentiles=[.01,.10,.25,.50,.75,.90,.99]).T"""),
C("""fig, axes = plt.subplots(1, 3, figsize=(12, 3))
portfolio['classic_fico'].hist(bins=25, ax=axes[0]); axes[0].set_title('FICO')
portfolio['current_ltv'].clip(upper=150).hist(bins=25, ax=axes[1]); axes[1].set_title('Current LTV')
portfolio['current_actual_upb'].hist(bins=25, ax=axes[2]); axes[2].set_title('Current UPB')
plt.tight_layout()"""),
M("Output: a checked modelling population with one loan identifier and ordered monthly observations.")
]),

"02_default_and_transitions.ipynb": make(
"Default and transition analysis",
"Default is the first 90+ DPD, REO or configured credit-related terminal event. The state matrix is a lifetime-PD benchmark.",
[
C('transitions = query("select * from transition_matrix"); transitions'),
C("""matrix = transitions.pivot(index='from_state', columns='to_state', values='transition_probability').fillna(0)
matrix"""),
C("""fig, ax = plt.subplots(figsize=(6, 4))
image = ax.imshow(matrix.values, vmin=0, vmax=1, cmap='Blues')
ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha='right')
ax.set_yticks(range(len(matrix.index)), matrix.index)
for i in range(len(matrix.index)):
    for j in range(len(matrix.columns)):
        ax.text(j, i, f'{matrix.iloc[i,j]:.1%}', ha='center', va='center')
plt.colorbar(image, ax=ax); plt.tight_layout()"""),
C('query("select completed_workout, count(*) defaults, avg(workout_lgd) average_lgd from default_workout_recovery group by completed_workout")'),
M("A cure is a move from a delinquent state to a lower-risk state in the following observed month. Only consecutive monthly records enter the matrix."),
M("Output: consistent default dates, default events, cure rates and roll rates.")
]),

"03_pd_woe_iv.ipynb": make(
"PD WOE and Information Value",
"WOE and IV are calculated on development data only. They are screening diagnostics for binary PD targets and are not used for LGD or EAD.",
[
M("WOE = log(distribution of non-defaults / distribution of defaults). IV is the sum of the distribution difference multiplied by WOE."),
C("woe = query('select * from pd_woe_iv')\nwoe.head()"),
C("""sample = query(\"select classic_fico, target, observation_weight from pd_development_sample where target_name='12_month_pd' and split='development'\")
sample['fico_band'] = pd.qcut(sample['classic_fico'], 10, duplicates='drop')
groups = sample.groupby('fico_band', observed=False).apply(
    lambda x: pd.Series({'defaults': (x.target*x.observation_weight).sum(),
                         'non_defaults': ((1-x.target)*x.observation_weight).sum()}),
    include_groups=False).reset_index()
epsilon = 0.5
groups['default_share'] = (groups.defaults+epsilon)/(groups.defaults.sum()+epsilon*len(groups))
groups['non_default_share'] = (groups.non_defaults+epsilon)/(groups.non_defaults.sum()+epsilon*len(groups))
groups['woe'] = np.log(groups.non_default_share/groups.default_share)
groups['iv_component'] = (groups.non_default_share-groups.default_share)*groups.woe
groups[['fico_band','defaults','non_defaults','woe','iv_component']]"""),
C("groups.iv_component.sum()"),
C("""variable_iv = (woe.groupby(['model','variable'], as_index=False)['variable_iv'].max()
.sort_values(['model','variable_iv'], ascending=[True,False]))
variable_iv.groupby('model').head(12)"""),
C("""selected = variable_iv.iloc[0]['variable']
woe[woe['variable'].eq(selected)][['model','variable','bin','count','default_rate','woe','iv_component','variable_iv']]"""),
M("IV is used with economic judgement, missing-value review, stability and correlation checks. It is not an automatic variable-selection rule.")
]),

"04_twelve_month_pd.ipynb": make(
"Twelve-month PD model",
"The target is first default within the next 12 months. The primary model is logistic regression.",
[
M("""Development

Data: surviving loan-month observations with a complete 12-month outcome window.

Features: borrower quality, leverage, delinquency history, modification, balance, rate and loan type.

Split: chronological development, calibration and out-of-time periods.

Method: imputation, standardisation, one-hot encoding, logistic regression and odds calibration.

Tests: ROC-AUC, Gini, KS, Brier score and observed versus predicted rates."""),
C("metrics = query(\"select * from pd_model_metrics where model='12-month logistic PD'\"); metrics"),
C("""pd_data = query(\"select * from pd_development_sample where target_name='12_month_pd'\")
numeric = ['classic_fico','original_dti','original_ltv','current_ltv','current_dpd','max_dpd_3m','max_dpd_6m','max_dpd_12m','current_actual_upb','current_interest_rate','modification_history','unemployment_rate','gdp_growth_yoy','hpi_growth_yoy']
categorical = ['occupancy_status','loan_purpose','property_type']
features = numeric + categorical
prepare = ColumnTransformer([
    ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),('scale', StandardScaler())]), numeric),
    ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical)
])
notebook_pd = Pipeline([('prepare', prepare),('model', LogisticRegression(max_iter=500, C=.5))])
development = pd_data[pd_data.split.eq('development')]
notebook_pd.fit(development[features], development.target, model__sample_weight=development.observation_weight)
rows = []
for name in ['development','out_of_time']:
    data = pd_data[pd_data.split.eq(name)]
    probability = notebook_pd.predict_proba(data[features])[:,1]
    rows.append({'sample':name,'rows':len(data),'defaults':int(data.target.sum()),
                 'weighted_auc':roc_auc_score(data.target, probability, sample_weight=data.observation_weight),
                 'weighted_brier':brier_score_loss(data.target, probability, sample_weight=data.observation_weight)})
pd.DataFrame(rows)"""),
C("""coefficients = query("select * from pd_model_coefficients where model='12-month PD'")
coefficients.reindex(coefficients.coefficient.abs().sort_values(ascending=False).index).head(15)"""),
C("""calibration = query("select * from pd_calibration where model='12-month logistic PD'")
calibration[['band','observations','defaults','observed_rate','predicted_rate','calibration_ratio']]"""),
C("""plot_data = calibration.reset_index(drop=True)
ax = plot_data[['observed_rate','predicted_rate']].plot(marker='o')
ax.set_xlabel('Risk band'); ax.set_ylabel('Default rate'); ax.set_title('Out-of-time calibration')"""),
M("Odds calibration multiplies raw logistic odds by one constant estimated on the calibration period, then converts the adjusted odds back to probability."),
C("""production_pd = joblib.load(ROOT / 'outputs' / 'models' / 'pd12_logistic.joblib')
raw_example = 0.01
raw_odds = raw_example/(1-raw_example)
calibrated_odds = raw_odds*production_pd.odds_multiplier
{'odds_multiplier':production_pd.odds_multiplier,
 'raw_pd':raw_example,
 'calibrated_pd':calibrated_odds/(1+calibrated_odds)}"""),
M("Output: current and origination 12-month PD estimates used in SICR and hazard calibration.")
]),

"05_lifetime_pd_hazard.ipynb": make(
"Lifetime PD hazard model",
"The target is default in the next month conditional on survival through the current month.",
[
M("Conditional PD is q(t). Survival is S(t-1). Marginal PD is S(t-1) x q(t). Cumulative PD is the sum of marginal PD through month T."),
C("query(\"select * from pd_model_metrics where model='discrete-time monthly hazard'\")"),
C("""hazard_data = query(\"select * from pd_development_sample where target_name='monthly_hazard'\")
hazard_numeric = ['classic_fico','original_dti','original_ltv','current_ltv','current_dpd','max_dpd_3m','max_dpd_6m','max_dpd_12m','current_actual_upb','current_interest_rate','modification_history','loan_age_sqrt','remaining_months_to_legal_maturity']
hazard_categorical = ['occupancy_status','loan_purpose','property_type']
hazard_features = hazard_numeric + hazard_categorical
hazard_prepare = ColumnTransformer([
 ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),hazard_numeric),
 ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),hazard_categorical)])
notebook_hazard = Pipeline([('prepare',hazard_prepare),('model',LogisticRegression(max_iter=500,C=.5))])
hazard_development = hazard_data[hazard_data.split.eq('development')]
notebook_hazard.fit(hazard_development[hazard_features], hazard_development.target,
                    model__sample_weight=hazard_development.observation_weight)
hazard_test = hazard_data[hazard_data.split.eq('out_of_time')]
hazard_probability = notebook_hazard.predict_proba(hazard_test[hazard_features])[:,1]
{'rows':len(hazard_test),'defaults':int(hazard_test.target.sum()),
 'weighted_auc':roc_auc_score(hazard_test.target,hazard_probability,sample_weight=hazard_test.observation_weight)}"""),
C("""example_q = np.array([0.010,0.015,0.020])
example_smm = 0.004
example_survival = np.r_[1.0, np.cumprod((1-example_q)*(1-example_smm))[:-1]]
example_mpd = example_survival*example_q
pd.DataFrame({'month':[1,2,3],'conditional_pd':example_q,'survival_at_start':example_survival,
              'prepayment_probability':example_smm,'marginal_pd':example_mpd,
              'cumulative_pd':example_mpd.cumsum()})"""),
M("The production model raises each raw monthly survival probability to a loan-specific exponent. The exponent is solved by bisection so Base marginal PD over months 1 to 12 equals the separate 12-month PD score. Prepayment remains a competing exit in survival."),
C("""term = query("select * from pd_term_structure where stage=2 and scenario='Base' and future_month<=120")
term.head(15)"""),
C("""fig, ax1 = plt.subplots()
ax1.plot(term.future_month, term.marginal_pd, label='Marginal PD')
ax1.set_xlabel('Future month'); ax1.set_ylabel('Marginal PD')
ax2 = ax1.twinx(); ax2.plot(term.future_month, term.cumulative_pd, color='tab:orange')
ax2.set_ylabel('Cumulative PD'); ax1.set_title('Stage 2 Base PD term structure')"""),
M("The near-perfect hazard ranking is treated cautiously because delinquency immediately before default is highly predictive. Target timing and leakage checks remain important.")
]),

"06_pd_macro_scenarios.ipynb": make(
"PD macro scenarios",
"A separate macroeconomic satellite adjusts the loan-level monthly hazard under Base, Upside and Downside paths.",
[
M("The satellite models monthly hazard log-odds residuals. Unemployment and the negative of GDP and HPI growth are constrained to non-negative coefficients. The production result is a monthly scenario term structure, not one lifetime percentage."),
C("query('select * from pd_macro_satellite_metrics')"),
C("""satellite = joblib.load(ROOT / 'outputs' / 'models' / 'pd_macro_satellite.joblib')
ridge = satellite.model.named_steps['ridge']
pd.DataFrame({'variable':['unemployment_rate','weak_gdp_growth','weak_hpi_growth'],
              'coefficient':ridge.coef_})"""),
C("macro = query(\"select * from macro_scenarios where month_number<=60\"); macro.head()"),
C("""fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for name, data in macro.groupby('scenario'):
    axes[0].plot(data.month_number, data.unemployment_rate, label=name)
    axes[1].plot(data.month_number, data.gdp_growth_yoy, label=name)
    axes[2].plot(data.month_number, data.hpi_growth_yoy, label=name)
axes[0].set_title('Unemployment'); axes[1].set_title('GDP growth'); axes[2].set_title('HPI growth')
axes[0].legend(); plt.tight_layout()"""),
C("""pd_scen = query("select * from pd_term_structure where stage=2 and future_month<=120")
for name, data in pd_scen.groupby('scenario'):
    plt.plot(data.future_month, data.cumulative_pd, label=name)
plt.xlabel('Future month'); plt.ylabel('Average cumulative PD'); plt.legend(); plt.title('Stage 2 PD scenarios')"""),
M("Output: conditional, survival, marginal and cumulative PD by loan, scenario and month.")
]),

"07_lgd_workout_eda.ipynb": make(
"Workout LGD EDA",
"LGD development begins with actual defaults, EAD at default and discounted net recoveries.",
[
C("""recoveries = query('select * from default_workout_recovery')
for column in ['ead_at_default','workout_lgd','recovery_duration_months','estimated_ltv_at_default']:
    recoveries[column] = pd.to_numeric(recoveries[column], errors='coerce')
recoveries.shape"""),
C("""recoveries.groupby('completed_workout').agg(
defaults=('loan_id','count'), ead=('ead_at_default','sum'),
average_lgd=('workout_lgd','mean'), median_lgd=('workout_lgd','median'),
average_duration=('recovery_duration_months','mean'))"""),
C("""recoveries.groupby(['cashflow_sign_convention','ead_at_default_source','recovery_cashflow_source'],
                           dropna=False).size().rename('defaults').reset_index()"""),
C("""recoveries.loc[recoveries.completed_workout.eq(1),'workout_lgd'].hist(bins=25)
plt.xlabel('Workout LGD'); plt.ylabel('Defaults'); plt.title('Completed-workout LGD')"""),
C("recoveries[['ead_at_default','estimated_ltv_at_default','workout_lgd','recovery_duration_months']].corr()"),
M("Incomplete workouts remain identified as model estimates and are not mixed silently with completed recoveries.")
]),

"08_lgd_model.ipynb": make(
"LGD model and term structure",
"Completed-workout LGD is modelled on a bounded scale and projected for each possible future default month.",
[
M("""Development

Target: discounted workout LGD.

Features: LTV, EAD, property value, location, property type, occupancy, modification, HPI, unemployment, insurance and loan age.

Split: first 80% of completed defaults by date for development and last 20% for out-of-time testing.

Method: logit-transform LGD, fit regularised ridge regression and apply the inverse-logit transformation.

Tests: MAE, RMSE, R-squared and observed versus predicted LGD."""),
C("query('select * from lgd_model_metrics')"),
C("""lgd_data = query('select * from lgd_development_sample where completed_workout=1 and workout_lgd is not null and ead_at_default>0')
lgd_data['default_date'] = pd.to_datetime(lgd_data.default_date)
lgd_numeric = ['estimated_ltv_at_default','ead_at_default','original_property_value','unemployment_rate','hpi_growth_yoy','mortgage_insurance_percentage','loan_age_at_default']
lgd_categorical = ['property_state','property_type','occupancy_status','modification_at_default']
dates = lgd_data.default_date.drop_duplicates().sort_values().reset_index(drop=True)
cutoff = dates.iloc[min(max(int(len(dates)*.8),1),len(dates)-1)]
lgd_development = lgd_data[lgd_data.default_date<cutoff]
lgd_test = lgd_data[lgd_data.default_date>=cutoff]
lgd_prepare = ColumnTransformer([
 ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),lgd_numeric),
 ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),lgd_categorical)])
y = np.log(lgd_development.workout_lgd.clip(.005,.995)/(1-lgd_development.workout_lgd.clip(.005,.995)))
notebook_lgd = Pipeline([('prepare',lgd_prepare),('ridge',Ridge(alpha=5))]).fit(lgd_development[lgd_numeric+lgd_categorical],y)
prediction = 1/(1+np.exp(-notebook_lgd.predict(lgd_test[lgd_numeric+lgd_categorical])))
{'development_rows':len(lgd_development),'out_of_time_rows':len(lgd_test),
 'out_of_time_mae':mean_absolute_error(lgd_test.workout_lgd,prediction),
 'median_benchmark_mae':mean_absolute_error(lgd_test.workout_lgd,np.repeat(lgd_development.workout_lgd.median(),len(lgd_test)))}"""),
C("""lgd_term = query("select * from lgd_term_structure where stage=2 and future_month<=120")
for name, data in lgd_term.groupby('scenario'):
    plt.plot(data.future_month, data.lgd, label=name)
plt.xlabel('Future month'); plt.ylabel('Average LGD'); plt.legend(); plt.title('Stage 2 LGD term structure')"""),
M("The fitted regression supplies the Base conditional LGD. Scenario HPI changes collateral value and the collateral shortfall floor. Output is LGD conditional on default in month t.")
]),

"09_ead_model.ipynb": make(
"EAD model and backtest",
"Mortgage EAD conditional on future default is based on contractual amortisation and an observed balance-at-default adjustment.",
[
M("""Development

Target: balance outstanding if default occurs in future month t.

Inputs: current UPB, contractual rate, remaining maturity and observed payoff behaviour.

Method: fixed-rate annuity balance, non-interest-bearing deferred UPB and an observed balance-at-default adjustment.

Tests: one-month forecast MAE, RMSE, MAPE, weighted absolute percentage error and mean error.

No CCF is used because there is no undrawn commitment."""),
C("json.loads((ROOT / 'outputs' / 'models' / 'ead_calibration.json').read_text())"),
C("query('select * from ead_backtest_metrics')"),
C("""ead_sample = query('select * from ead_development_sample')
ead_sample['error'] = ead_sample.predicted_default_balance-ead_sample.actual_default_balance
ead_sample.groupby('split').agg(rows=('loan_id','count'),mae=('error',lambda x:x.abs().mean()),mean_error=('error','mean'))"""),
C("""principal, annual_rate, remaining, month = 200000, 5.0, 360, 12
r = annual_rate/100/12
payment = principal*r/(1-(1+r)**(-remaining))
balance = principal*(1+r)**month-payment*((1+r)**month-1)/r
{'monthly_payment':payment,'balance_after_12_payments':balance}"""),
C("""ead = query("select * from ead_profile where stage in (1,2)")
for stage, data in ead.groupby('stage'):
    plt.plot(data.future_month, data.ead, label=f'Stage {stage}')
plt.xlabel('Future month'); plt.ylabel('Portfolio EAD'); plt.legend(); plt.title('EAD profile')"""),
M("Output: EAD by loan and future month.")
]),

"10_ead_macro_test.ipynb": make(
"EAD macro sensitivity test",
"A loan-only prepayment model is compared with a loan-plus-macro model on an out-of-time sample.",
[
C("ead_macro = query('select * from ead_macro_sensitivity'); ead_macro"),
C("""query('''
select future_month, max(ead)-min(ead) scenario_ead_difference
from ead_term_structure
where stage=2
group by future_month
order by future_month
limit 24
''')"""),
M("This test concerns voluntary-prepayment ranking. It does not directly model the balance conditional on default. If the macro variables do not add material ranking power, the production EAD remains scenario-invariant and scenario sensitivity enters PD and LGD instead. The displayed scenario mortgage-rate path does not override contractual EIR discounting or conditional EAD.")
]),

"11_scenarios_and_weights.ipynb": make(
"Scenarios and weights",
"Base, Upside and Downside paths are internally generated assumptions. IFRS 9 does not prescribe their names or probabilities.",
[
C("weights = query('select * from scenario_weight_analysis'); weights"),
C("""blend = CFG['scenario_weighting']['configured_weight_share']
weights[['scenario','configured_weight','historical_analogue_frequency','selected_weight']].assign(
recalculated_weight=lambda x: blend*x.configured_weight +
                            (1-blend)*x.historical_analogue_frequency,
difference=lambda x: x.recalculated_weight-x.selected_weight)"""),
C("""query('''select scenario,
avg(case when month_number<=12 then unemployment_rate end) unemployment_first_year,
avg(case when month_number<=12 then gdp_growth_yoy end) gdp_first_year,
avg(case when month_number<=12 then hpi_growth_yoy end) hpi_first_year,
avg(case when month_number<=12 then mortgage_rate end) mortgage_rate_first_year
from macro_scenarios group by scenario''')"""),
M("Production weights blend the configured judgement and historical analogue frequency in equal proportions. The result is a modelling choice, not an official forecast or an IFRS 9 requirement. The mortgage-rate paths provide context; contractual loan rates remain the EIR approximation and EAD is scenario-invariant."),
C("query('select * from scenario_summary order by scenario')")
]),

"12_sicr_and_staging.ipynb": make(
"SICR and staging",
"Staging combines lifetime-PD deterioration, delinquency, the 30-DPD backstop, modification history and default.",
[
M("Stage 3 is assigned first. Stage 2 is assigned when any configured SICR trigger applies. Remaining loans are Stage 1. Current and origination lifetime PD are calculated from the monthly hazard over the same remaining horizon, so the comparison measures risk deterioration rather than a horizon difference."),
C("""{'relative_lifetime_pd_multiple':CFG['sicr']['relative_pd_multiple'],
 'absolute_lifetime_pd_increase':CFG['sicr']['absolute_pd_increase'],
 'dpd_backstop':CFG['sicr']['dpd_backstop']}"""),
C("query('select * from stage_summary order by stage')"),
C("""query('''select stage, primary_stage_reason, count(*) loans,
sum(current_actual_upb) exposure
from reporting_date_portfolio
group by stage, primary_stage_reason
order by stage, loans desc''')"""),
M("Stage 1 uses defaults in the next 12 months. Stage 2 uses remaining lifetime. Stage 3 uses discounted recovery cash shortfall.")
]),

"13_projection_cube_and_ecl.ipynb": make(
"Projection cube and ECL",
"This notebook joins the monthly PD, LGD, EAD and discount-factor terms in the production calculation.",
[
M("Period ECL = Marginal PD x LGD x EAD x Discount Factor. Scenario ECL is summed first. Loan ECL is the weighted sum of complete scenario ECL values."),
C("""loan = query("select loan_id from worked_trace_summary where stage=2").iloc[0,0]
loan"""),
C("""detail = query(f'''select loan_id,stage,scenario,scenario_weight,future_month,
conditional_pd,survival_probability,marginal_pd,cumulative_pd,
ead,projected_property_value,projected_ltv,lgd,discount_factor,period_ecl
from worked_trace_monthly where loan_id='{loan}' order by scenario,future_month''')
detail.head(24)"""),
C("""detail['recalculated_ecl'] = detail.marginal_pd * detail.lgd * detail.ead * detail.discount_factor
(detail.period_ecl-detail.recalculated_ecl).abs().max()"""),
C("""detail.groupby(['scenario','scenario_weight'],as_index=False).period_ecl.sum()"""),
C("""scenario_totals = detail.groupby(['scenario','scenario_weight'],as_index=False).period_ecl.sum()
scenario_totals['weighted_contribution'] = scenario_totals.period_ecl*scenario_totals.scenario_weight
scenario_totals[['scenario','period_ecl','scenario_weight','weighted_contribution']], scenario_totals.weighted_contribution.sum()"""),
M("SQLite views provide separate Base, Upside and Downside outputs while ecl_projection_cube remains the authoritative calculation table.")
]),

"14_worked_loan_traces.ipynb": make(
"Worked loan traces",
"One Stage 1, Stage 2 and Stage 3 loan is traced from monthly terms to final provision.",
[
C("summary = query('select * from worked_trace_summary order by stage'); summary"),
C("""monthly = query('select * from worked_trace_monthly')
monthly.groupby(['loan_id','stage','scenario','scenario_weight'], as_index=False).period_ecl.sum()"""),
C("""reconciliation = (monthly.groupby(['loan_id','stage','scenario','scenario_weight'], as_index=False).period_ecl.sum()
.assign(weighted_ecl=lambda x: x.period_ecl*x.scenario_weight)
.groupby(['loan_id','stage'], as_index=False).weighted_ecl.sum()
.merge(summary[['loan_id','loan_ecl']], on='loan_id'))
reconciliation['difference'] = reconciliation.weighted_ecl-reconciliation.loan_ecl
reconciliation"""),
C("""stage1_id = summary.loc[summary.stage.eq(1),'loan_id'].iloc[0]
monthly[(monthly.loan_id.eq(stage1_id)) & (monthly.scenario.eq('Base'))][
['future_month','marginal_pd','lgd','ead','discount_factor','period_ecl']].head(12)"""),
C("""stage2_id = summary.loc[summary.stage.eq(2),'loan_id'].iloc[0]
monthly[(monthly.loan_id.eq(stage2_id)) & (monthly.scenario.eq('Base'))][
['future_month','conditional_pd','survival_probability','marginal_pd','lgd','ead','discount_factor','period_ecl']].head(18)"""),
C("""stage3_id = summary.loc[summary.stage.eq(3),'loan_id'].iloc[0]
monthly[monthly.loan_id.eq(stage3_id)][
['scenario','ead','projected_property_value','gross_collateral_proceeds','recovery_expenses',
 'expected_mi_recovery','expected_recovery','discount_factor','discounted_expected_recovery','period_ecl']]"""),
M("Stage 3 has no performing-loan marginal PD. Its provision is exposure less discounted scenario-specific expected recovery.")
])
}


def build(execute=False):
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        path = NOTEBOOK_DIR / name
        if execute:
            NotebookClient(
                nb, timeout=600, kernel_name="python3",
                resources={"metadata": {"path": str(ROOT)}}
            ).execute()
        nbf.write(nb, path)
        print(path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    build(args.execute)
