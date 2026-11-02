"""
LLM evaluation metrics: perplexity, BLEU, ROUGE, and LLM-as-a-judge.

The lesson: n-gram metrics (BLEU/ROUGE) measure *word overlap* with a reference,
not *meaning*. Three candidate answers are scored against one reference:

  paraphrase   - same meaning, different words   -> low BLEU/ROUGE, but correct
  near_copy    - nearly the reference wording    -> high BLEU/ROUGE, correct
  wrong_fact   - reference wording, one key fact flipped -> high BLEU/ROUGE, WRONG

An LLM judge reads for meaning and should rank wrong_fact last.

Run from this folder:   python 12LLMReview.py
The judge step calls the Claude API and needs credentials (ANTHROPIC_API_KEY or
`ant auth login`) plus `pip install anthropic pydantic`; without them it is skipped.
"""
import logging
import math
import warnings

logging.getLogger("torchao").setLevel(logging.ERROR)  # transformers imports torchao; its GPU-only kernels fail to load on CPU

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

REFERENCE = "The quick brown fox jumps over the lazy dog."
CANDIDATES = {
    "paraphrase": "A fast brown fox leaps across a tired dog.",
    "near_copy": "The quick brown fox jumped over the lazy dog.",
    "wrong_fact": "The lazy brown dog jumps over the quick fox.",
}
JUDGE_MODEL = "claude-opus-5-5"


# --- 1. Perplexity (intrinsic: needs no reference) --------------------------
def pick_device():
    """Use the GPU only if this torch build ships kernels for it (recent CUDA wheels
    dropped older cards such as the GTX 10xx series, which then crash on first use)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch warns about unsupported GPUs here
        if not torch.cuda.is_available():
            return "cpu"
        major, minor = torch.cuda.get_device_capability(0)
        supported = f"sm_{major}{minor}" in torch.cuda.get_arch_list()
    if not supported:
        print(f"GPU {torch.cuda.get_device_name(0)} (sm_{major}{minor}) is not supported "
              f"by torch {torch.__version__}; using CPU.")
        return "cpu"
    return "cuda"


def load_lm(model_id="gpt2"):
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()
    return tokenizer, model, device


def perplexity(text, tokenizer, model, device):
    """exp(mean next-token cross-entropy). Lower = text is less 'surprising' to the model.
    It measures fluency/typicality, not correctness: a fluent false statement can score well."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        loss = model(ids, labels=ids).loss  # labels are shifted internally
    return math.exp(loss.item())


# --- 2. Overlap metrics (need a human reference) ----------------------------
_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def overlap_scores(prediction, reference):
    ref_tok, pred_tok = reference.lower().split(), prediction.lower().split()
    # Unsmoothed BLEU-4 is exactly 0 for any short sentence missing a 4-gram match,
    # which is why sentence-level BLEU is almost always reported with smoothing.
    with warnings.catch_warnings():  # nltk warns on each zero score; the zero is the point here
        warnings.simplefilter("ignore", UserWarning)
        bleu_raw = sentence_bleu([ref_tok], pred_tok)
    bleu = sentence_bleu([ref_tok], pred_tok, smoothing_function=_smooth)
    r = _rouge.score(reference, prediction)
    return {"bleu_raw": bleu_raw, "bleu": bleu,
            "rouge1": r["rouge1"].fmeasure, "rougeL": r["rougeL"].fmeasure}


# --- 3. LLM-as-a-judge -----------------------------------------------------
JUDGE_PROMPT = """You are grading a candidate sentence against a reference sentence.
Score 1-10 for how well the candidate preserves the reference's MEANING (who does what to whom,
and the descriptive facts). Wording differences do not matter; changed facts matter a lot.

Reference: {reference}
Candidate: {candidate}"""


def make_judge():
    """Return judge(candidate, reference) -> (score, reason), or None if unavailable."""
    try:
        import anthropic
        from pydantic import BaseModel, Field
    except ImportError:
        print("Judge skipped: pip install anthropic pydantic")
        return None

    class Verdict(BaseModel):
        reason: str = Field(description="One sentence explaining the score")
        score: int = Field(description="1 (meaning lost) to 10 (meaning identical)")

    client = anthropic.Anthropic()  # credentials are resolved (and checked) on the first call

    def judge(candidate, reference):
        response = client.beta.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=2000,
            # If the judge model declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_format=Verdict,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                reference=reference, candidate=candidate)}],
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return None, f"no verdict (stop_reason={response.stop_reason})"
        v = response.parsed_output
        return v.score, v.reason

    return judge


def main():
    tokenizer, model, device = load_lm()
    judge = make_judge()

    print(f"Reference: {REFERENCE}")
    print(f"Reference perplexity: {perplexity(REFERENCE, tokenizer, model, device):.1f}\n")
    header = f"{'candidate':<11} {'PPL':>6} {'BLEU raw':>9} {'BLEU sm':>8} {'ROUGE-1':>8} {'ROUGE-L':>8} {'judge':>6}"
    print(header)
    print("-" * len(header))

    reasons = {}
    for name, text in CANDIDATES.items():
        ppl = perplexity(text, tokenizer, model, device)
        s = overlap_scores(text, REFERENCE)
        score = "-"
        if judge is not None:
            try:
                score, reasons[name] = judge(text, REFERENCE)
            except TypeError as e:  # no credentials configured, or an anthropic SDK older than 2.x
                print(f"Judge skipped ({e}).\n  Set ANTHROPIC_API_KEY (or run `ant auth login`) "
                      "and make sure `anthropic~=2.3` is installed.")
                judge = None
            except Exception as e:  # network/auth/rate-limit: keep the classroom demo running
                print(f"Judge call failed ({e.__class__.__name__}: {e}); continuing without it.")
                judge = None
        print(f"{name:<11} {ppl:>6.1f} {s['bleu_raw']:>9.3f} {s['bleu']:>8.3f} "
              f"{s['rouge1']:>8.3f} {s['rougeL']:>8.3f} {score!s:>6}")

    for name, reason in reasons.items():
        print(f"  {name}: {reason}")

    print("\nTakeaways:")
    print(" * wrong_fact keeps every reference word (ROUGE-1 = 1.0) yet reverses who jumps over whom:")
    print("   overlap metrics cannot see meaning. ROUGE-L (longest common subsequence) does drop for it.")
    print(" * paraphrase is correct but scores lowest on BLEU/ROUGE: one reference cannot cover all valid wordings.")
    print(" * Perplexity needs no reference and tracks fluency, not truth.")
    print(" * An LLM judge reads for meaning, but it costs money and latency, can be biased (e.g. toward")
    print("   longer answers or its own style), and should be spot-checked against human labels.")


if __name__ == "__main__":
    main()
