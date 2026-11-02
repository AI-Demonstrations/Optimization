# Optimization
Code to optimize models for both performance and resource usage

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
All notebooks run on CPU; a GPU speeds up 7, 7B and 9. Run notebooks from this folder so relative `Data/` paths resolve.
Notebooks 4, 6, 8 and `12LLMReview.py` download small Hugging Face models (GPT-2, OPT-125m, a sentence-embedding model), and 9 downloads ImageNet MobileNetV2 weights, on first run.

For the optional LLM-as-a-judge step in `12LLMReview.py`, set `ANTHROPIC_API_KEY` (or run `ant auth login`); without credentials the script skips that step.

## Metrics methods across various model types
| Notebook | Topic |
|---|---|
| `1_W_SupervisedMetricsDiabetesMetrics` | Binary classification metrics, class imbalance, threshold choice; regression metrics |
| `1B_W_SupervisedTensorFlowMetrics` | Multi-class metrics and one-vs-rest ROC (Iris) |
| `2_W_UnsupervisedMetrics` | Clustering metrics and choosing k |
| `3_W_ReinforcementLearning` | Q-learning on FrozenLake: reward, success rate, sample efficiency, robustness |
| `3B_W_ReinforcementLearning` | Q-learning with a continuous state (CartPole) via discretization |
| `4_W_GenerativeMetrics` | Perplexity, BLEU, ROUGE, semantic similarity, generation speed |
| `12LLMReview.py` | Script: perplexity, BLEU and ROUGE vs. an LLM judge on a paraphrase, a near-copy and a fact-flipped sentence; shows overlap metrics miss meaning (judge step optional, uses the Claude API) |

## Measuring cost
| Notebook | Topic |
|---|---|
| `5_W_MeasuringTimeBasic` | Timing training and prediction correctly |
| `10_W_MeasuringTime` | Latency percentiles, throughput vs. batch size, memory, profiling |

## Model Adaptation via Transfer Learning and Model Quantization
| Notebook | Topic |
|---|---|
| `6_W_CompressLLMs` | Pruning, low-rank factorization, knowledge distillation |
| `7_W_ComplextoSimpleModelCompare` | Complex vs. simplified CNN: accuracy vs. size and latency |
| `7B_W_SimpleDeepReduction` | Systematic CNN reduction (global pooling, separable convs, width multiplier) |
| `8_W_QuantizedLLM` | Quantizing a transformer and a pre-trained LLM: size, perplexity, latency |
| `9_W_TransferLearning` | Transfer learning and fine-tuning with MobileNetV2 (cats vs. dogs) |

## Data
| Path | Used by |
|---|---|
| `Data/diabetes.csv` | 1 |
| `Data/llm_eval_text.txt` | 6, 8 (held-out text for perplexity) |
| `Data/train/`, `Data/validation/` | 9 (Kaggle Dogs vs. Cats subset) |

`archive/` holds early drafts superseded by the numbered notebooks; see `archive/README.md`.
