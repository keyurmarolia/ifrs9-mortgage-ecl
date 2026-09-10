# Core data dictionary

## Required architecture tables

- `origination_mortgages`: one row per loan with initial borrower, collateral and contractual terms plus retained original property-value estimate and `synthetic_field_flag`.
- `monthly_loan_performance`: one row per real loan-month with balance, delinquency, modification, terminal status, default event/date and recovery fields.
- `default_workout_recovery`: one row per default with EAD, EAD source, sign-normalised recovery source, observed and remaining discounted recoveries, uncapped and LGD-capped total recoveries, completion flags, actual-loss reconciliation, duration and workout LGD.
- `macroeconomic_history`: official monthly unemployment/HPI and forward-filled quarterly real GDP with derived yearly growth, configured availability lags and a revised-history limitation flag in `source_status`.
- `reporting_date_portfolio`: active loan population with current/origination PD, lifetime-PD comparison, every staging trigger, stage and horizon.
- `ecl_projection_cube`: Loan × Scenario × FutureMonth audit table.

## Projection cube fields

- Identity: `loan_id`, `stage`, `scenario`, `scenario_weight`, `future_month`, `period`.
- PD: `raw_conditional_pd`, `conditional_pd`, `survival_probability`, `marginal_pd`, `cumulative_pd`, `hazard_calibration_exponent`.
- EAD: `contractual_balance`, `ead`, `ead_method`.
- Collateral/LGD: `projected_property_value`, `projected_ltv`, `lgd`, `lgd_method`.
- Discount/ECL: `discount_factor`, `discount_rate_method`, `period_ecl`, `calculation_method`.
- Macro: `unemployment_rate`, `gdp_growth_yoy`, `hpi_growth_yoy`, `mortgage_rate`, `hpi_factor`.

Stage 3 rows intentionally have null conditional/survival/marginal/cumulative PD fields because their `period_ecl` is a discounted recovery cash shortfall, not a performing-loan probability calculation.

## Summary export units

In `executive_summary.csv`, coverage, PD and LGD are decimal ratios. Multiply by 100 to express a percentage. For example, 0.0007870576 corresponds to 0.07870576%. Excel applies percentage display formats separately.
