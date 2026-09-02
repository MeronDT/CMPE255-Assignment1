export interface ZoneInfo {
  location_id: number;
  zone: string;
  borough: string;
  service_zone: string;
  lat: number;
  lon: number;
}

export interface HistoricalStats {
  n_trips: number;
  avg_duration_min: number;
  avg_fare_amount: number;
  avg_trip_distance_mi: number;
}

export interface PredictResponse {
  predicted_duration_min: number;
  predicted_fare_amount: number;
  haversine_km: number;
  pickup_zone: string;
  dropoff_zone: string;
  pickup_borough: string;
  dropoff_borough: string;
  is_rush_hour: boolean;
  is_airport_trip: boolean;
  historical: HistoricalStats | null;
}

export interface SearchTrial {
  trial: number;
  params: Record<string, number>;
  val_rmse: number;
  val_mae: number;
  val_r2: number;
  best_iteration: number;
  train_seconds: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface TargetReport {
  target: string;
  baseline_val_metrics: { rmse: number; mae: number; r2: number };
  search_trials: SearchTrial[];
  best_params: Record<string, number>;
  best_val_rmse: number;
  test_metrics: { rmse: number; mae: number; r2: number };
  feature_importance: FeatureImportance[];
}

export interface TournamentResult {
  backbone: string;
  rmse: number;
  mae: number;
  r2: number;
  train_seconds: number;
}

export interface FeatureTransformResult {
  variant: string;
  rmse: number;
  r2: number;
  n_features: number;
}

export interface HillClimbStep {
  iteration: number;
  params: Record<string, number>;
  rmse: number;
  move: string;
}

export interface BlendResult {
  weight_a: number;
  rmse: number;
  r2: number;
}

export interface PaperBenchmark {
  source: string;
  r2: number | null;
  note: string;
}

export interface AutoResearchTarget {
  target: string;
  search_sample_size: number;
  phase1_tournament: { results: TournamentResult[]; winning_backbone: string };
  phase2_feature_transform: { results: FeatureTransformResult[]; winning_variant: string };
  phase3_hill_climbing: { seed_params: Record<string, number>; path: HillClimbStep[]; best_params: Record<string, number>; best_metrics: { rmse: number; r2: number } };
  phase4_blending: { results: BlendResult[]; best: BlendResult };
  final: { winner_stage: string; rmse: number; r2: number };
  paper_benchmark: PaperBenchmark;
  feature_transform_added: string;
}

export interface AutoResearchFinalizeEntry {
  autoresearch_params: Record<string, number>;
  previous_test_rmse: number;
  full_data_retrain_test_metrics: { rmse: number; mae: number; r2: number };
  redeployed: boolean;
}

export interface AutoResearchResponse {
  history: {
    generated_at: string;
    methodology: string;
    targets: Record<string, AutoResearchTarget>;
  };
  finalize: Record<string, AutoResearchFinalizeEntry>;
}

export interface ModelInfo {
  eda: {
    n_rows_raw: number;
    n_columns: number;
    date_range_claimed: string;
    missing_values_pct: Record<string, number>;
    data_quality_issues: { issue: string; count: number }[];
    target_stats_raw: Record<string, { mean: number; median: number; std: number; p99: number; max: number; min: number }>;
    categorical_cardinality: Record<string, number>;
  };
  data_preparation: {
    n_rows_raw: number;
    n_rows_after_cleaning: number;
    pct_dropped: number;
    n_train: number;
    n_test: number;
    cutoff_day_of_month: number;
    feature_cols: string[];
    target_cols: string[];
    n_od_pairs_with_history: number;
  };
  model_search: {
    generated_at: string;
    targets: Record<string, TargetReport>;
  };
}
