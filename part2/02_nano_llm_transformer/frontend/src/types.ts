export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  generation_ms: number;
  tokens_generated: number;
  tokens_per_sec: number;
}

export interface SystemInfo {
  device: string;
  python_version: string;
  torch_version: string;
  gpu_name?: string;
  gpu_total_memory_mb?: number;
  gpu_allocated_memory_mb?: number;
  cuda_version?: string;
  bf16_supported?: boolean;
}

export interface TrainStep {
  iter: number;
  train_loss: number;
  val_loss: number;
  val_perplexity: number;
  lr: number;
  elapsed_sec: number;
  gpu_mem_mb: number;
  tokens_per_sec?: number;
}

export interface TrainHistory {
  run_tag: string;
  config: Record<string, unknown>;
  n_params: number;
  device: string;
  max_iters: number;
  steps: TrainStep[];
  total_train_seconds: number;
}

export interface GenerationSample {
  prompt: string;
  completion: string;
}

export interface EvaluationSummary {
  base: {
    final_train_loss: number;
    final_val_loss: number;
    final_val_perplexity: number;
    total_train_seconds: number;
    n_params: number;
    samples: GenerationSample[];
  };
  sft: {
    final_train_loss: number;
    final_val_loss: number;
    final_val_perplexity: number;
    total_train_seconds: number;
    samples: GenerationSample[];
  };
}

export interface ModelInfo {
  eda: {
    tinystories: {
      n_stories: number;
      total_chars: number;
      avg_chars_per_story: number;
      avg_words_per_story: number;
      max_words_per_story: number;
      unique_words_sample_5000_stories: number;
      top_20_words: [string, number][];
    };
    alpaca: {
      n_examples: number;
      n_with_input_field: number;
      avg_instruction_words: number;
      avg_output_words: number;
      max_output_words: number;
    };
  };
  data_preparation: {
    vocab_size: number;
    context_length: number;
    pretrain: { n_train_tokens: number; n_val_tokens: number; n_stories: number };
    sft: { n_train_examples: number; n_val_examples: number; n_available_total: number };
  };
  train_history_base: TrainHistory;
  train_history_sft: TrainHistory;
  evaluation: EvaluationSummary;
}

export interface ArchTournamentResult {
  variant: string;
  val_loss: number;
  val_perplexity: number;
  n_params: number;
  seconds: number;
}

export interface ShapeSearchResult {
  shape: string;
  d_model: number;
  n_layers: number;
  n_heads: number;
  val_loss: number;
  val_perplexity: number;
  n_params: number;
  seconds: number;
}

export interface HillClimbStep {
  iteration: number;
  params: Record<string, number>;
  val_loss: number;
  move: string;
}

export interface AutoResearchHistory {
  generated_at: string;
  methodology: string;
  proxy_budget: { iters: number; batch_size: number; context_length: number };
  phase1_architecture_tournament: { results: ArchTournamentResult[]; winner: string };
  phase2_shape_search: { results: ShapeSearchResult[]; winner: string; winner_dims: Record<string, number> };
  phase3_hill_climbing: { seed_params: Record<string, number>; path: HillClimbStep[]; best_params: Record<string, number>; best_val_loss: number };
  phase4_model_soup: { model_a_val_loss: number; model_b_val_loss: number; soup_val_loss: number; soup_beats_both: boolean };
  final_recommended_config: Record<string, number>;
}

export interface AutoResearchResponse {
  history: AutoResearchHistory | null;
  finalize: Record<string, unknown> | null;
}
