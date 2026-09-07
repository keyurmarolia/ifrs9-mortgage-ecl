# Methodology

## Data and default

The model keeps one origination row per loan and one performance row per loan-month. Default begins on the first 90+ DPD observation, REO status or configured credit-related terminal event. Consecutive monthly states form the transition matrix.

## Twelve-month PD

The target is first default within the next 12 months. The model is logistic regression with chronological development, calibration and out-of-time periods. Features cover borrower quality, leverage, delinquency, modification, balance, rate and loan characteristics. WOE and IV are development-sample screening diagnostics; they do not replace economic judgement or model testing.

## Lifetime PD and macroeconomic satellite

The discrete-time hazard target is default next month conditional on survival through the current month. The loan model excludes macroeconomic variables. A separate non-negative ridge satellite explains monthly hazard log-odds residuals using unemployment, weak GDP growth and weak HPI growth.

For future month `t`:

`MPD(t) = S(t-1) × q(t)`

`CPD(T) = sum of MPD(1) through MPD(T)`

The Base first-12-month marginal PD sum is calibrated to the loan's 12-month PD. The same calibration exponent applies to all scenarios. Scenario variation enters once through the satellite.

## LGD

Historical workout LGD equals EAD at default less discounted net recoveries, divided by EAD at default. Completed workouts fit a logit-transformed ridge model using a chronological development and out-of-time split. Incomplete workouts retain a model-estimate flag and use estimated remaining recovery.

The fitted model provides the Base conditional LGD. Scenario HPI changes projected property value and the collateral shortfall floor for every possible default month.

## EAD

Conditional mortgage EAD uses the fixed-rate amortization balance, non-interest-bearing deferred UPB and an observed balance-at-default adjustment. Prepayment is included as a competing risk in survival, not as a reduction to EAD conditional on default. No CCF is used because there is no undrawn commitment.

The observed macroeconomic prepayment test did not improve out-of-time ranking materially, so EAD is scenario-invariant.

## Discounting and staging

The current contractual mortgage rate is the EIR approximation:

`DF(t) = 1 / (1 + EIR/12)^t`

Stage 3 is assigned first. Stage 2 uses quantitative lifetime-risk deterioration, the 30-DPD backstop, modification and prior-default indicators. Remaining accounts are Stage 1. Stage 1 uses default events in months 1 to 12, Stage 2 uses remaining lifetime, and Stage 3 uses discounted expected recovery cash shortfall.

## ECL and scenario weighting

For performing loans:

`Period ECL(i,t,s) = MPD(i,t,s) × LGD(i,t,s) × EAD(i,t) × DF(i,t)`

Monthly values are summed within each complete scenario. Loan ECL is the weighted sum of scenario ECL values. PD, LGD and EAD are not weighted separately.

The scenario weights blend configured judgement and historical analogue frequency. Scenario names and probabilities are model assumptions, not IFRS 9 requirements.
