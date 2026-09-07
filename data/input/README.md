# Input files

The pipeline expects the following local files:

```text
data/input/freddie_mac/sample_2015/sample_orig_2015.txt
data/input/freddie_mac/sample_2015/sample_perf_2015.txt
data/input/fred/UNRATE.csv
data/input/fred/GDPC1.csv
data/input/fhfa/hpi_po_monthly_hist.xlsx
```

Freddie Mac files use the pipe-delimited Single-Family Loan-Level Dataset layout. `UNRATE.csv` and `GDPC1.csv` use the standard FRED date-and-value export layout. The FHFA workbook is the monthly purchase-only HPI history.

Source data is kept outside version control. File locations can be changed in `config/project.yaml`.
