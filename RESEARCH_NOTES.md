# ELVIS Benchmark — Research Notes

## Goal

Evaluate a group of Vision Language Models (VLMs) on the ELVIS Gestalt benchmark and investigate whether general model capability correlates with Gestalt perceptual ability. The focus is on multiple diverse models, not a single model family.

---

## Problems With the Current Evaluation

### 1. Heavy Priming
The current protocol shows the model 3 positive + 3 negative labeled training images and the principle name before any test image is shown. This means the model does not need to perceive the Gestalt pattern — it only needs to find visual similarity to the provided examples.

### 2. Two-Stage Rule Verbalization Bias
Stage 1 asks the model to explicitly articulate a logic rule. A model that perceives proximity correctly but cannot verbalize it will fail Stage 2 even though its perception is sound. This penalizes models with weak language-visual grounding, not weak Gestalt perception.

### 3. Binary Classification Conflates Ability With Guessing
50% accuracy is trivially achievable. Accuracy/F1 on binary tasks provides limited signal about perceptual sensitivity.

### 4. Classification ≠ Perception
The task asks "does this fit the rule?" not "what do you see?". A model with no Gestalt perceptual ability can still succeed through few-shot analogical reasoning.

---

## How Humans Are Evaluated (Literature)

### Dot Lattice Paradigm — Kubovy & Wagemans (1995); Kubovy, Holcombe & Wagemans (1998)
- Present dot arrays for ~300ms (prevents deliberate reasoning)
- Ask: "which orientation do you perceive?" — no examples, no labels
- Measure: log-odds of perceived orientation as a function of relative distance
- Produces a continuous **sensitivity slope** (person-specific), not a binary score
- Key finding: Pure Distance Law — grouping by proximity decays exponentially with relative distance

### Repetition Discrimination Task (RDT) — Palmer & Beck (2007)
- Participants do a secondary task (detect two identical targets in an array)
- Grouping is never mentioned — it is measured indirectly via reaction times
- Within-group targets are found faster than between-group targets
- Objective, implicit, no strategic response bias possible
- Validated for proximity, color similarity, common region, element connectedness

### Detection vs. Grouping Thresholds (Vision Research, 2010)
- Measure separately: minimum stimulus difference for detection vs. for perceived grouping
- Finding: grouping threshold is 5–7× higher than detection threshold
- Shows that perceiving a difference and perceiving a group are dissociable abilities

---

## Key Differences: Human Evaluation vs. ELVIS

| Dimension | Human | ELVIS (current) |
|---|---|---|
| Prior examples shown | None | 3 positive + 3 negative |
| Principle name given | Never | Always (except `_no_principle`) |
| Requires verbalization | No | Yes (Stage 1) |
| Task awareness | Often none (implicit) | Explicit |
| Response format | Forced-choice orientation, RT | Binary positive/negative |
| Metric | Sensitivity slope, d-prime, threshold | Accuracy, F1 |
| Exposure duration | ~300ms (forced pre-attentive) | Unlimited |

---

## Evaluation Modes Proposed

| | Mode 1: Baseline | Mode 2: ZS Named | Mode 3: ZS Blind |
|---|---|---|---|
| Examples shown | 3 pos + 3 neg | None | None |
| Principle name given | Yes | Yes | No |
| Stage 1 (rule induction) | Yes | No | No |
| Prompt | Infer rule from examples | "Does this image exhibit {principle}?" | "Are objects grouped into distinct clusters?" |
| Answer format | positive / negative | yes / no | yes / no |
| What it measures | Few-shot conceptual classification | Concept knowledge + visual classification | Closest to pure visual perception |
| Model key suffix | *(existing)* | `_zs_named` | `_zs_blind` |

Running all three per model produces a **priming gap** (Mode 1 − Mode 3) which itself may correlate with model capability — stronger models should rely less on priming.

---

## Implementation Plan

### Scope
All changes on a single branch (fork: `maplesyrup-0606/ELVIS`). No separate branches — modes are differentiated by model key in `evaluate_models.py`.

### New Model Keys

| Model Key | Base Checkpoint | Mode | Slurm GPU |
|---|---|---|---|
| `internVL_zs_named` | InternVL3-2B | ZS Named | `1g.10gb` |
| `internVL_zs_blind` | InternVL3-2B | ZS Blind | `1g.10gb` |
| `internVL_8B_zs_named` | InternVL3-8B | ZS Named | `2g.20gb` |
| `internVL_8B_zs_blind` | InternVL3-8B | ZS Blind | `2g.20gb` |
| `internVL_14B_zs_named` | InternVL3-14B | ZS Named | `3g.40gb` |
| `internVL_14B_zs_blind` | InternVL3-14B | ZS Blind | `3g.40gb` |
| `internVL_38B_zs_named` | InternVL3-38B | ZS Named | `h100:1` |
| `internVL_38B_zs_blind` | InternVL3-38B | ZS Blind | `h100:1` |
| `internVL_X_zs_named` | InternVL3-78B | ZS Named | `h100:2` |
| `internVL_X_zs_blind` | InternVL3-78B | ZS Blind | `h100:2` |
| `llava_zs_named` | LLaVA-OneVision-7B | ZS Named | `2g.20gb` |
| `llava_zs_blind` | LLaVA-OneVision-7B | ZS Blind | `2g.20gb` |

### Files to Change
- `scripts/baseline_models/conversations.py` — add 4 new prompt functions (2 InternVL, 2 LLaVA)
- `scripts/baseline_models/internVL.py` — add `evaluate_llm_zeroshot` helper + runners per size
- `scripts/baseline_models/llava.py` — add `evaluate_llm_zeroshot` helper + 2 runners
- `scripts/evaluate_models.py` — register all new keys

### Key Design Decisions
- Stage 1 (rule induction) skipped entirely in zero-shot modes — no training images loaded
- Response parsing: `yes` → 1, `no` → 0, anything else → logged as **ambiguous** (excluded from metrics, counted per run)
- WandB: shared project per principle (e.g. `ELVIS-InternVL-proximity`) — all modes logged together for direct comparison
- Output: `/elvis_result/{principle}/zeroshot/{date}/` — dated subfolder separates zero-shot from baseline results
- Standard `run_*(data_path, img_size, principle, batch_size, device, img_num, epochs, start_num, task_num)` signature preserved
- InternVL size variants reuse the same loader pattern, only the HuggingFace checkpoint path changes

### Slurm Job Structure
- One job per model per principle (5 principles × N models)
- GPU tier matched to model size (MIG for ≤14B, full H100 for ≥38B)
- Cluster: Fir (fir.alliancecan.ca), Alliance Canada at SFU

---

## Deferred Directions (Not Yet)

- **Linear probing of frozen representations** — tests what the visual encoder knows independent of the language head; requires model internals (locks out API models); best for mechanistic follow-up
- **Attention weight analysis** — interesting for understanding *where* Gestalt information is processed; requires model internals
- **Threshold-based evaluation** — vary Gestalt cue strength across a range, find per-model sensitivity curves; requires generating graded stimulus variants
- **Forced-choice orientation task** — closest human analogue; requires restructuring response format and potentially dataset
- **Model capacity correlation study** — plot ELVIS score vs. MMMU/MMStar scores across a diverse model set; feasible once a clean zero-shot metric exists

---

## Models of Interest (Free / Open Source)

| Model | Size | Notes |
|---|---|---|
| InternVL3-2B | 2B | Already in codebase |
| LLaVA-OneVision-7B | 7B | Already in codebase |
| InternVL3-8B | 8B | Easy add — same loader |
| InternVL3-78B | 78B | Already in codebase as `internVL_X` |
| Qwen2.5-VL-7B | 7B | Not yet in codebase |
| DeepSeek-VL2-small | ~3B | In codebase, not registered |

---

## References

- Kubovy, M. & Wagemans, J. (1995). Grouping by proximity and multistability in dot lattices. *Psychological Science*, 6, 225–234.
- Kubovy, M., Holcombe, A.O. & Wagemans, J. (1998). On the lawfulness of grouping by proximity. *Cognitive Psychology*, 35, 71–98.
- Palmer, S.E. & Beck, D. (2007). The repetition discrimination task: An objective method for studying perceptual grouping. *Perception & Psychophysics*, 69, 68–78.
- Detection vs. grouping thresholds (2010). *Vision Research*. doi:10.1016/j.actpsy.2003.06.003
- Wagemans et al. (2012). A century of Gestalt psychology in visual perception. *PMC3482144*.
- Neural decoding dissociates perceptual grouping (2021). *bioRxiv* 2021.11.15.468580.
