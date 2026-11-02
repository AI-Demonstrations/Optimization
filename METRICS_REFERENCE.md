# Metrics Reference

Companion to [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md); the same data is in [metrics_reference.csv](metrics_reference.csv). For each metric: how it is calculated, what good and bad values look like, and the result measured in the course notebooks. Course results come from the notebooks' saved outputs.

**Contents:** [Supervised (Classification)](#supervised-classification) · [Supervised (Regression)](#supervised-regression) · [Unsupervised (Clustering)](#unsupervised-clustering) · [Reinforcement Learning](#reinforcement-learning) · [Generative Models (Language)](#generative-models-language) · [Generative Models (GANs)](#generative-models-gans) · [Generative Models (VAEs)](#generative-models-vaes) · [Resource Use (any model)](#resource-use-any-model)

## Supervised (Classification)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Clearly above the majority-class baseline | At or below the majority-class baseline (always predicting the most common class) | Diabetes 97.2% vs. 91.5% majority baseline; Iris 96.7% (30 test rows, so one error = 3.3 points) *(notebook 1, 1B)* |
| **Precision** | `TP / (TP + FP)` | Close to 1 | Close to 0 | Diabetes 99.7% at threshold 0.5 *(notebook 1, 1B)* |
| **Recall (Sensitivity)** | `TP / (TP + FN)` | Close to 1 | Close to 0 | Diabetes 67.6%: misses 551 of 1,700 diabetic patients despite 97.2% accuracy *(notebook 1, 1B)* |
| **F1-score** | `2 * (Precision * Recall) / (Precision + Recall)` | Close to 1 | Close to 0 | Diabetes 0.806; Iris macro F1 0.967 *(notebook 1, 1B)* |
| **Confusion Matrix and Decision Threshold** | Count TP, FP, FN, TN at threshold t; expected cost = C_FP * FP + C_FN * FN. With calibrated probabilities the cost-minimizing t = C_FP / (C_FP + C_FN). | Threshold that minimizes expected business cost | Default 0.5 used without checking the cost of each error type | Threshold 0.5: 3 false alarms, 551 missed cases. Threshold 0.1: 1,613 false alarms, 180 missed. 0.1 wins if a miss costs more than ~4.3 follow-up tests. *(notebook 1)* |
| **Macro vs. Weighted Averaging (multi-class)** | Macro = unweighted mean of per-class scores; Weighted = per-class scores weighted by class support | Macro and weighted both high and close together | Macro well below weighted (a rare class is failing) | Iris (balanced): macro precision 0.970, recall 0.967, F1 0.967; macro and weighted are equal *(notebook 1B)* |
| **ROC-AUC** | Area under the TPR (Recall) vs. FPR curve; one-vs-rest for multi-class | Close to 1 | Close to 0.5 (random ranking) | Iris one-vs-rest macro AUC 0.997 *(notebook 1, 1B)* |

<details><summary>How to improve, advantages and disadvantages</summary>

**Accuracy**

- *How to improve:* Address class imbalance (resampling, class weights); tune hyperparameters.
- *Advantages:* Intuitive and easy to understand. Fine for balanced datasets.
- *Disadvantages:* Highly misleading under class imbalance: always predicting 'no' scores 91.5% on the diabetes data while catching no cases. Always report the baseline beside it.

**Precision**

- *How to improve:* Raise the decision threshold; improve features.
- *Advantages:* Use when false positives are costly (e.g., spam filtering).
- *Disadvantages:* Ignores false negatives. Usually falls as recall rises.

**Recall (Sensitivity)**

- *How to improve:* Lower the decision threshold; class weights or resampling for the rare class.
- *Advantages:* Use when false negatives are costly (e.g., disease screening).
- *Disadvantages:* Ignores false positives. Usually falls as precision rises.

**F1-score**

- *How to improve:* Balance precision and recall (threshold choice); feature engineering.
- *Advantages:* Single number that balances precision and recall; more informative than accuracy on imbalanced data.
- *Disadvantages:* Weights precision and recall equally, which rarely matches real costs. Less intuitive than accuracy.

**Confusion Matrix and Decision Threshold**

- *How to improve:* Sweep thresholds and pick by cost; calibrate probabilities first.
- *Advantages:* Shows exactly which errors occur. Turns metric choice into an explicit business decision.
- *Disadvantages:* Needs an estimate of what each error costs. Not a single score, so harder to compare models.

**Macro vs. Weighted Averaging (multi-class)**

- *How to improve:* Improve the weak class: more data, class weights, targeted features.
- *Advantages:* Extends precision, recall and F1 to many classes. Macro exposes poor minority-class performance.
- *Disadvantages:* Weighted averaging can hide a failing rare class. The two diverge only on imbalanced data.

**ROC-AUC**

- *How to improve:* Better features or model; imbalance-handling methods.
- *Advantages:* Threshold-invariant. Measures how well the model ranks positives above negatives.
- *Disadvantages:* Can look overly optimistic on highly imbalanced data (use precision-recall AUC there). Says nothing about calibration or the chosen threshold.

</details>

## Supervised (Regression)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **MSE** | `(1/n) * Σ(y_i - ŷ_i)^2` | Close to 0 | Near or above the MSE of always predicting the mean | Reported with RMSE and MAE on the diabetes regression task *(notebook 1)* |
| **RMSE** | `sqrt( (1/n) * Σ(y_i - ŷ_i)^2 )` | Close to 0 | Near the target's standard deviation (no better than the mean) | Diabetes regression RMSE 53.9 (target units) *(notebook 1)* |
| **MAE** | `(1/n) * Σ\|y_i - ŷ_i\|` | Close to 0 | Near the MAE of always predicting the median | Diabetes regression MAE 42.8 *(notebook 1)* |
| **R-squared** | `1 - (Σ(y_i - ŷ_i)^2 / Σ(y_i - ȳ)^2)` | Close to 1 | 0 or negative (no better than predicting the mean) | Diabetes R² 0.453 vs. −0.012 for a mean-predictor baseline *(notebook 1)* |

<details><summary>How to improve, advantages and disadvantages</summary>

**MSE**

- *How to improve:* Handle outliers; improve features and scaling.
- *Advantages:* Heavily penalizes large errors. Differentiable, so it is the usual training loss.
- *Disadvantages:* Very sensitive to outliers. Squared units are hard to interpret.

**RMSE**

- *How to improve:* Same as MSE.
- *Advantages:* Same ranking as MSE but in the target's own units, so easier to explain.
- *Disadvantages:* Still sensitive to outliers.

**MAE**

- *How to improve:* Handle outliers that the model fits poorly; improve features; train with an MAE/Huber loss if MAE is the target metric.
- *Advantages:* Robust to outliers. In the target's original units.
- *Disadvantages:* Not differentiable at zero. Does not penalize large errors more heavily.

**R-squared**

- *How to improve:* Add or remove features; check for multicollinearity and nonlinearity.
- *Advantages:* Scale-free. Interpretable as the share of variance explained.
- *Disadvantages:* Rises as predictors are added (use adjusted R-squared). Does not show bias in predictions.

</details>

## Unsupervised (Clustering)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Silhouette Score** | (b - a) / max(a, b), averaged over points (a = mean intra-cluster distance, b = mean distance to nearest other cluster) | Close to 1 | Close to 0 (overlapping) or negative (misassigned) | 0.79 on synthetic blobs; agrees with Davies-Bouldin that k = 4 is best *(notebook 2)* |
| **Calinski-Harabasz Index** | `[Trace(B_k) / (k - 1)] / [Trace(W_k) / (n - k)]` | High (compare across k on the same data) | Low relative to other k | 3386.4 on synthetic blobs *(notebook 2)* |
| **Davies-Bouldin Index** | `(1/k) * Σ max_{j≠i} ((σ_i + σ_j) / d(c_i, c_j))` | Close to 0 | High positive value | 0.29 on synthetic blobs *(notebook 2)* |
| **Inertia (K-Means, within-cluster sum of squares)** | `Σ_{i=1}^{k} Σ_{x ∈ C_i} \|\|x - μ_i\|\|^2` | Use the elbow: the k after which inertia stops dropping sharply | Choosing k by lowest inertia (it always falls as k grows) | Falls steadily with k; used only with the elbow, alongside silhouette and Davies-Bouldin *(notebook 2)* |
| **Adjusted Rand Index (ARI)** | Agreement between cluster assignments and true labels over all point pairs, corrected for chance | Close to 1 | Close to 0 (chance level) or negative | 0.99 on synthetic blobs (possible only because true labels exist) *(notebook 2)* |

<details><summary>How to improve, advantages and disadvantages</summary>

**Silhouette Score**

- *How to improve:* Adjust number of clusters (k); scale features; try other algorithms.
- *Advantages:* Bounded between -1 and 1. Measures cohesion and separation. Needs no labels.
- *Disadvantages:* O(N^2) on large datasets. Misleading for non-convex or density-based clusters.

**Calinski-Harabasz Index**

- *How to improve:* Adjust k; try different distance metrics or algorithms.
- *Advantages:* Fast to compute. Needs no labels.
- *Disadvantages:* Unbounded, so only comparable on the same data. Favors convex, spherical clusters.

**Davies-Bouldin Index**

- *How to improve:* Adjust k; try different algorithms.
- *Advantages:* Simple, uses only the data and cluster assignments. Needs no labels.
- *Disadvantages:* Unbounded, so absolute values are hard to compare. Assumes convex clusters.

**Inertia (K-Means, within-cluster sum of squares)**

- *How to improve:* Adjust k; scale features; try different algorithms.
- *Advantages:* Easy to compute and intuitive. Drives the elbow method.
- *Disadvantages:* Always decreases with k, so it cannot choose k alone. Not normalized; depends on data scale. Favors spherical clusters.

**Adjusted Rand Index (ARI)**

- *How to improve:* Same as other clustering metrics.
- *Advantages:* Direct check against ground truth. Corrected for chance.
- *Disadvantages:* Needs true labels, which real unsupervised problems lack. Usable only on synthetic or labeled benchmark data.

</details>

## Reinforcement Learning

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Cumulative Reward (Return)** | Σ_{t=0}^{T} γ^t * R_t, averaged over evaluation episodes | High positive value (e.g., 500 = solved on CartPole) | Low or negative value | CartPole: best checkpoint averages 500 (solved), but the final Q-table averages 96. Keep the best policy, as with early stopping. *(notebook 3, 3B)* |
| **Episode Length** | Count of time steps T until a terminal state | Task-dependent: low for goal-reaching tasks; high for survival tasks (e.g., CartPole) | The opposite | FrozenLake greedy policy reaches the goal in the optimal 6 steps *(notebook 3, 3B)* |
| **Success Rate** | `Successful Episodes / Total Episodes` | Close to 1 | Close to 0 | FrozenLake: 88.9% during training, 100% for the greedy policy *(notebook 3)* |
| **Robustness (Performance under Environment Shift)** | Same metric (success rate or return) measured on a changed environment the agent did not train on | Close to the in-distribution score | Large drop from the in-distribution score | FrozenLake policy: 100% on the training lake, 3.9% on the slippery lake *(notebook 3)* |

<details><summary>How to improve, advantages and disadvantages</summary>

**Cumulative Reward (Return)**

- *How to improve:* Tune learning rate (α), discount (γ) and exploration schedule; refine the reward; keep the best checkpoint.
- *Advantages:* Direct measure of the RL objective.
- *Disadvantages:* Susceptible to reward hacking with a poorly designed or sparse reward. Noisy during training because of exploration.

**Episode Length**

- *How to improve:* Refine the reward (e.g., a step penalty for goal tasks).
- *Advantages:* Good efficiency measure for tasks that need quick completion.
- *Disadvantages:* Meaning flips between goal-reaching and survival tasks.

**Success Rate**

- *How to improve:* More exploration early; tune learning stability; evaluate the greedy policy, not training episodes.
- *Advantages:* Interpretable bottom-line metric for goal-oriented tasks.
- *Disadvantages:* Sparse binary signal, hard to optimize early in training. Training success rate includes exploration noise.

**Robustness (Performance under Environment Shift)**

- *How to improve:* Train with environment randomization; use stochastic training environments.
- *Advantages:* The RL equivalent of a held-out test set. Shows whether the policy generalizes.
- *Disadvantages:* Requires designing realistic variations. Results depend on which shifts you choose.

</details>

## Generative Models (Language)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Perplexity** | `exp(- (1/N) * Σ log P(w_i))` | Low value (relative to other texts or models with the same tokenizer) | High value | GPT-2's own rambling scores lowest (19). OPT-125m baseline 31.7; also the quality check for pruning and quantization. *(notebook 4, 6, 8, 12LLMReview.py)* |
| **BLEU** | Geometric mean of modified n-gram precisions (n = 1..4) × brevity penalty; use smoothing for single sentences | Close to 1 | Close to 0 | Paraphrase 0.06 vs. fact-flipped sentence 0.17 (smoothed): the wrong answer scores higher *(notebook 4, 12LLMReview.py)* |
| **ROUGE (ROUGE-1, ROUGE-L)** | ROUGE-1 = unigram overlap F1; ROUGE-L = longest-common-subsequence F1 | Close to 1 | Close to 0 | Fact-flipped sentence: ROUGE-1 = 1.0, ROUGE-L = 0.556; correct paraphrase ROUGE-1 = 0.333 *(notebook 4, 12LLMReview.py)* |
| **Semantic Similarity (Embedding Cosine)** | cos(e_candidate, e_reference) from a sentence-embedding model | Close to 1 | Close to 0 | Paraphrase 0.65 vs. 0.01 for unrelated text (its BLEU is only 0.06) *(notebook 4)* |
| **LLM-as-a-Judge** | Prompt a strong LLM with a rubric to score each output (e.g., 1-10) or compare two outputs | High scores that agree with human labels | Scores that disagree with human spot checks | Expected to rank the fact-flipped sentence last, which BLEU and ROUGE fail to do (needs an API key) *(`12LLMReview.py`)* |

<details><summary>How to improve, advantages and disadvantages</summary>

**Perplexity**

- *How to improve:* Larger model; more or cleaner training data; domain fine-tuning.
- *Advantages:* Standard intrinsic NLP metric. Needs no reference text.
- *Disadvantages:* Measures predictability, not correctness: a fluent false statement can score well. Not comparable across tokenizers or vocabularies.

**BLEU**

- *How to improve:* Generate closer to the reference wording; use several references.
- *Advantages:* Cheap and standard for machine translation. Reproducible.
- *Disadvantages:* Rewards word overlap, not meaning: penalizes valid paraphrases. Unsmoothed sentence BLEU is 0 for any short sentence with no 4-gram match.

**ROUGE (ROUGE-1, ROUGE-L)**

- *How to improve:* Same as BLEU.
- *Advantages:* Standard for summarization. ROUGE-L is sensitive to word order.
- *Disadvantages:* Overlap, not meaning: a sentence with its facts swapped can score perfectly.

**Semantic Similarity (Embedding Cosine)**

- *How to improve:* Same as other reference metrics; choose an embedding model suited to the domain.
- *Advantages:* Credits paraphrases that overlap metrics miss. Cheap once the embedding model is loaded.
- *Disadvantages:* Depends on the embedding model. Can rate a sentence with one fact flipped as highly similar.

**LLM-as-a-Judge**

- *How to improve:* Explicit rubric; structured output; swap answer order; average repeated calls; use a judge from a different model family.
- *Advantages:* Reads for meaning, catches factual errors overlap metrics miss. Scales better than human review.
- *Disadvantages:* Costs money and latency per call. Biased toward longer, confident or same-style answers and toward one position; scores vary between runs. Validate against human labels (e.g., Spearman or Cohen's kappa).

</details>

## Generative Models (GANs)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Inception Score** | `exp(E_x [KL(p(y\|x) \|\| p(y))])` | High value | Low value | Not covered in the course notebooks |
| **FID** | `\|\|μ_r - μ_g\|\|^2 + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})` | Low value | High value | Not covered in the course notebooks |

<details><summary>How to improve, advantages and disadvantages</summary>

**Inception Score**

- *How to improve:* Tune generator/discriminator architectures; stabilize training (e.g., spectral normalization).
- *Advantages:* Correlates with human judgment of image quality and diversity.
- *Disadvantages:* Ignores the real data distribution. Fooled by adversarial images; misses mode collapse within classes.

**FID**

- *How to improve:* Address mode collapse; more or better training data.
- *Advantages:* More robust than Inception Score. Compares generated images to real images directly.
- *Disadvantages:* Assumes Gaussian features. Computationally intensive; biased for small sample sizes.

</details>

## Generative Models (VAEs)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Reconstruction Error** | `MSE(x, x') or Binary Cross-Entropy(x, x')` | Low value | High value | Not covered in the course notebooks |

<details><summary>How to improve, advantages and disadvantages</summary>

**Reconstruction Error**

- *How to improve:* Adjust the latent dimension (z); balance reconstruction vs. KL divergence terms.
- *Advantages:* Directly measures how well the model encodes and decodes its input.
- *Disadvantages:* Can produce blurry images (especially with MSE). Does not capture perceptual realism.

</details>

## Resource Use (any model)

| Metric | Calculation | Good | Bad | Course result |
| --- | --- | --- | --- | --- |
| **Latency (mean and p95/p99)** | Wall-clock time per request with time.perf_counter(), after warm-up, over many repeats; report mean ± std and percentiles. On GPU, synchronize before reading the clock. | Tail (p95/p99) within the service-level target | Only a single run or the mean reported; tail above target | Mean 0.077 ms, p99 0.101 ms, max 0.191 ms. One prediction ~246 µs vs. ~0.39 µs per row in a batch of 1,000 (632× faster). *(notebook 5, 10)* |
| **Throughput** | Samples (or tokens) processed per second at a given batch size | High value at an acceptable latency | Low value, or batch size chosen without measuring | 13,000 → 80,000 samples/s by batch 32, then flat (dips at 512) *(notebook 10)* |
| **Memory and Model Size** | Parameters × bytes per parameter for weights (file size on disk) plus activation memory at the chosen batch size | Fits the target device with headroom | Exceeds device memory, or size estimated without measuring | OPT-125m: 501 MB fp32 → 247 MB int8 weight-only (embeddings stay fp32). Activation memory 0.03 MB at batch 1 → 16.8 MB at batch 512. *(notebook 8, 10, 6)* |
| **Parameter Count** | Σ elements of all trainable weight tensors (model.summary() / sum(p.numel())) | Smallest model that meets the quality target | Reductions measured by layer count instead of parameters | First Dense layer holds 72.8% of 7B's weights; global average pooling removes ~70% of parameters for 0.5 points of accuracy *(notebook 7, 7B, 6)* |
| **Quality Retained after Compression** | Quality metric (accuracy or perplexity) of the compressed model vs. the original, reported with its size and speed | Small quality change for a large size or speed gain | Large quality loss, or quality not re-measured after compressing | int8 weight-only perplexity 31.53 vs. 31.66 fp32; 50% pruning 136.6; distilled 2,410-param student 94.6% vs. teacher 97.4% *(notebook 6, 7, 7B, 8, 9)* |

<details><summary>How to improve, advantages and disadvantages</summary>

**Latency (mean and p95/p99)**

- *How to improve:* Batch requests; smaller or quantized model; faster hardware; remove per-call overhead.
- *Advantages:* Matches what users experience. Percentiles capture the tail that users notice.
- *Disadvantages:* Meaningless without the method (clock, warm-up, repeats, batch size, hardware). Noisy; the minimum of repeats best reflects the code's own cost.

**Throughput**

- *How to improve:* Increase batch size until throughput levels off; use hardware-friendly data types.
- *Advantages:* Right metric for offline and batch jobs.
- *Disadvantages:* Rising batch size also raises per-batch latency. Not monotonic, so measure rather than assume.

**Memory and Model Size**

- *How to improve:* Quantization (fewer bytes per parameter); fewer parameters; smaller batches.
- *Advantages:* Determines which devices can run the model and how many copies fit per server.
- *Disadvantages:* Pruning zeroes weights without shrinking the file unless sparse storage is used. Parts not quantized (e.g., embeddings) limit the savings.

**Parameter Count**

- *How to improve:* Profile where parameters sit; global average pooling; separable convolutions; width multipliers; distillation.
- *Advantages:* Hardware-independent measure of model size; locates where to cut.
- *Disadvantages:* Fewer parameters do not guarantee lower latency: small models are dominated by per-call overhead.

**Quality Retained after Compression**

- *How to improve:* Apply one technique at a time; fine-tune after pruning; distill; prefer weight-only quantization.
- *Advantages:* Produces the trade-off table needed for model selection.
- *Disadvantages:* Results depend on hardware and data. Must be re-measured for each technique and setting.

</details>
