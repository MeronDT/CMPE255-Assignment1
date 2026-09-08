# Model Card: NBA MVP Vote-Share Champion and Challenger

## Model details

- **Task:** Predict continuous MVP `award_share` and rank candidates within each season.
- **Champion:** weighted `HistGradientBoostingRegressor`.
- **Required challenger:** Extra Trees hurdle model: probability of receiving votes multiplied by predicted positive vote share.
- **Training for locked evaluation:** 1982–2018.
- **Locked test:** 2019–2022.
- **Production artifact:** refit through 2022 after evaluation.
- **Eligibility cohort:** at least 100 total minutes; repeat reporting at 500 minutes.

Champion hyperparameters are frozen: learning rate 0.08, 31 maximum leaf nodes, L2 regularization 1.0, 200 iterations, minimum 20 samples per leaf, early stopping disabled, and random state 42. Training weights are `1 + 5 × award_share`; final predictions are clipped to `[0, 1]`.

## Intended use

- Educational analysis of historical MVP voting.
- Producing a statistical candidate ranking for a complete regular season.
- Comparing model rankings with actual voting after results become available.
- Supporting—not replacing—expert discussion of award races.

## Out-of-scope use

- Claiming who objectively “deserved” an award.
- Causal claims about what makes a player valuable.
- Betting, personnel, compensation, or other high-stakes decisions.
- Scoring partial seasons without clearly labeling the distribution shift.
- Combining outputs from different seasons into one global rank.

## Evaluation summary

On 2019–2022, the champion produced RMSE 0.0263 over all eligible players, recipient RMSE 0.1553, R² 0.8033, NDCG@5 0.8934, mean winner rank 1.25, and 75% top-1 accuracy. The hurdle challenger produced slightly weaker regression but higher NDCG@5 (0.9219) and better season-total calibration.

## Known risks

- **Small test:** four seasons and only two unique winners.
- **Sparse target:** 96.29% of raw rows have zero vote share.
- **Winner underprediction:** champion mean bias for actual winners is −0.2170.
- **Boundary behavior:** 84.31% of champion test predictions require clipping, mostly tiny negative values mapped to zero.
- **Missing context:** injuries, games missed at key times, narrative, media attention, teammate competition, conference context, and ballot rules are not modeled.
- **Era and policy drift:** scoring environments and post-2023 eligibility rules can change the candidate pool.
- **Human-label subjectivity:** historical votes reflect human preferences and may encode persistent biases.

## Monitoring

Score exactly one complete season at a time. Validate schema and logical ranges before prediction. Monitor candidate count, missingness, feature PSI, champion/challenger top-1 agreement, top-five overlap, output clipping, season-total share, and—after official voting—RMSE, NDCG@5, and winner rank.

Do not automatically retrain. Trigger review after a schema change or two consecutive completed seasons with material degradation. Preserve the frozen historical result when adding new seasons.

## Artifact integrity

- File: `models/nba_mvp_production_models_through_2022.joblib`
- Size: 15,969,016 bytes in the verified build
- SHA-256: `e8980fb3ecb592005692fc26040589b9a9f48c412b2e91b84148661c77206b22`
- Verified conditions: deserializes successfully; trained through 2022; `award_share` absent from model features; target-free scoring test passes.

Joblib files can execute Python code during loading. Only load this artifact from a trusted copy of the repository and a compatible Python environment.
