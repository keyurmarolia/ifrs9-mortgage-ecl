# Assumptions and limitations

- The reporting date is 1 March 2026.
- The model sample contains 15,000 Freddie Mac loans; 3,471 are active with positive balances at the reporting date.
- Source Freddie Mac, FRED and FHFA observations are not synthetic.
- A separate official mortgage-rate history was not present in the supplied macro files. The portfolio contractual rate is the starting proxy for the displayed scenario-rate paths. Those paths do not replace loan-specific contractual EIRs or change conditional EAD.
- Base, Upside and Downside paths are internally generated assumptions.
- Selected scenario weights are 51.5159% Base, 31.7604% Upside and 16.7237% Downside. They blend configured judgement and historical analogue frequencies equally.
- The contractual mortgage rate approximates EIR because fee-level effective-yield data is unavailable.
- The lifetime hazard has unusually high discrimination because delinquency immediately before default is strongly predictive. Its target timing is next-month default conditional on survival.
- The PD satellite retained unemployment; weak GDP and HPI coefficients were zero after constrained regularisation. Its out-of-time monthly aggregate R-squared is -19.27% on 24 defaults, so it is used as a constrained scenario sensitivity rather than presented as a strongly validated predictive model.
- The completed-workout LGD sample is modest. Its out-of-time MAE is 3.79%, slightly above the 3.54% median benchmark MAE, and its out-of-time R-squared is negative. The term structure should therefore be read together with the benchmark and collateral shortfall floor.
- Incomplete workouts use explicitly flagged estimated remaining recoveries.
- EAD is scenario-invariant because macro variables did not materially improve the out-of-time prepayment test.
- Current unresolved Stage 3 recovery timing uses the historical completed-workout duration with workout age considered.
- The large Stage 2 share is sensitive to the aged 2015 cohort and the configured two-times relative or two-percentage-point absolute lifetime-PD deterioration thresholds.
- Current revised FRED and FHFA histories are used. Publication lags and a reporting-date cutoff are applied, but point-in-time vintage revisions cannot be reconstructed from the supplied files.
- This is a modelling implementation, not an adopted accounting policy or an official economic forecast.
