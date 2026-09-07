# IFRS 9 Mortgage ECL Model

This project calculates loan-level expected credit loss for a residential mortgage portfolio. It uses Freddie Mac loan data, FRED unemployment and real GDP data, and FHFA house-price data.

The main calculation is:

`Period ECL = Marginal PD × LGD × EAD × Discount Factor`

Each macroeconomic scenario is calculated in full before its probability weight is applied. Stage 3 uses a discounted recovery cash-shortfall calculation instead of performing-loan PD.

## Model sequence

1. Read origination, monthly performance and macroeconomic data.
2. Apply the 90+ DPD and credit-event default definition.
3. Build loan-month features and delinquency transitions.
4. Fit the 12-month logistic PD and monthly hazard models.
5. Fit the PD macroeconomic satellite and construct three scenarios.
6. Score current and origination risk and assign IFRS 9 stages.
7. Fit workout LGD and project scenario collateral values.
8. Build conditional mortgage EAD and loan-specific discount factors.
9. Create the Loan × Scenario × Future Month projection cube.
10. Calculate scenario, loan, stage and portfolio ECL.
11. Write SQLite, CSV and Excel outputs.

## Notebooks

The repository includes executed notebooks with saved outputs and anonymized account references. Read them in numerical order. They show data review, WOE and IV, chronological model fitting and testing, macroeconomic calibration, staging, term structures, the projection equation, and worked Stage 1, Stage 2 and Stage 3 accounts.

After generating them, open them with:

```bash
.venv/bin/python -m jupyterlab notebooks
```

## Run

Place the source files described in `data/input/README.md` in the input folders. Then run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_pipeline.py
.venv/bin/python scripts/build_notebooks.py --execute
```

## Main outputs

- `database/ifrs9_ecl.sqlite3`: local calculation database
- `outputs/IFRS_9_Mortgage_ECL_Model.xlsx`: reporting workbook
- `outputs/tables/account_level_ecl.csv`: anonymized account report
- `outputs/tables/worked_trace_monthly.csv`: monthly worked calculations

The authoritative detailed table is `ecl_projection_cube`. SQLite views provide separate Base, Upside and Downside projections and detailed PD, LGD, EAD and discount-factor terms.

## Current model run

- Reporting date: 1 March 2026
- Active mortgages: 3,471
- Gross exposure: $432,963,771.97
- Probability-weighted ECL: $424,001.87
- Coverage ratio: 0.0979%
- Stage 1: 192 loans
- Stage 2: 3,261 loans
- Stage 3: 18 loans

The saved run reports 25 passing internal controls; these are project checks, not production validation. Scenario ECL before probability weighting is $410,509 for Upside, $419,659 for Base and $458,859 for Downside.

Raw loan files and the local SQLite database are excluded from version control. The included notebooks, workbook and account-level reporting outputs use stable `RM` aliases. A fresh clone requires the source files listed in `data/input/README.md` before the pipeline can be rerun.
