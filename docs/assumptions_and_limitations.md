# Assumptions and limitations

- The reporting date is 1 March 2026.
- The model sample contains 15,000 Freddie Mac loans; 3,471 are active with positive balances at the reporting date.
- Source Freddie Mac, FRED and FHFA observations are not synthetic.
- A separate official mortgage-rate history was not present in the supplied macro files. The portfolio contractual rate is used as the scenario rate proxy.
- Base, Upside and Downside paths are internally generated assumptions.
- Selected scenario weights are 57.9805% Base, 25.0852% Upside and 16.9343% Downside. They blend configured judgement and historical analogue frequencies equally.
- The contractual mortgage rate approximates EIR because fee-level effective-yield data is unavailable.
- The lifetime hazard has unusually high discrimination because delinquency immediately before default is strongly predictive. Its target timing is next-month default conditional on survival.
- The PD satellite retained unemployment; weak GDP and HPI coefficients were zero after constrained regularisation.
- The completed-workout LGD sample is modest. Its out-of-time MAE is 4.56%, slightly above the 4.12% median benchmark MAE, so the model should be read with the benchmark.
- Incomplete workouts use explicitly flagged estimated remaining recoveries.
- EAD is scenario-invariant because macro variables did not materially improve the out-of-time prepayment test.
- Current unresolved Stage 3 recovery timing uses the historical completed-workout duration with workout age considered.
- The large Stage 2 share is sensitive to the aged 2015 cohort and the configured SICR thresholds.
- This is a modelling implementation, not an adopted accounting policy or an official economic forecast.
