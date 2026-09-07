# Data sources

## Freddie Mac

Single-Family Loan-Level Dataset, 2015 sample origination and monthly performance files. The model uses contractual terms, monthly balances, delinquency, modifications, terminal events and recovery fields.

Source: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

## FRED

- `UNRATE`: Civilian Unemployment Rate
- `GDPC1`: Real Gross Domestic Product

Source: https://fred.stlouisfed.org/

Quarterly real GDP is carried forward between releases before year-over-year growth is calculated. Model availability lags are one month for unemployment, three months for GDP and two months for HPI. The configured reporting-date cutoff prevents post-reporting observations from entering the model.

## FHFA

Purchase-Only House Price Index, monthly national history.

Source: https://www.fhfa.gov/data/hpi

## Publication boundary

Raw source files and the local calculation database are not part of the repository. The included workbook and CSV outputs contain aggregate results or anonymized account aliases.

The supplied FRED and FHFA files contain currently revised historical series rather than point-in-time vintages. Publication lags are applied, but later historical revisions cannot be removed without vintage source files. This limitation is recorded in the model outputs.
