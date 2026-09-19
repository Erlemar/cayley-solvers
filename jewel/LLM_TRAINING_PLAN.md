# Training a Christopher's Jewel Language Model

## A four-week, single-A100 research and implementation plan

**Status:** proposed  
**Hardware budget:** one NVIDIA A100 80 GB for up to four weeks  
**Primary objective:** build a genuine causal language model that accepts an English description of a Christopher's Jewel state and responds with one legal English move, then replans after the user reports the next state  
**Secondary objective:** determine whether the language model adds solving value beyond the existing structured Transformer, PDBs, exact ball, and interactive search

---

## 1. Executive decision

The project is feasible. The recommended system is not a free-form model that is trusted to simulate the puzzle. It is a neuro-symbolic conversational planner:

1. A deterministic parser converts the user's English colour description into a validated symbolic state.
2. A pretrained causal language model consumes the normalized English request and predicts one of twelve legal move tokens.
3. Auxiliary heads predict action regret, distance intervals, and whether search is needed.
4. The exact Jewel engine applies every move and remains the only source of truth for state transitions.
5. Low-confidence decisions invoke the existing PDB-guided interactive search and exact depth-8 endgame.
6. A deterministic verifier rejects every unparseable or non-solving returned path.

This is a real language model: it is pretrained on language, trained with a causal language-model objective, accepts paraphrased English dialogue, and emits natural-language responses. It is also deliberately constrained where symbolic correctness matters.

The recommended primary base is [`Qwen/Qwen3-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-0.6B-Base): a 0.6B-parameter causal language model with an Apache-2.0 license and a 32,768-token context window. The scale-up candidate is [`Qwen/Qwen3-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-1.7B-Base). The 0.6B model should be fully fine-tuned for the final run; LoRA should be used for cheap architecture and data ablations.

The project has two independent definitions of success:

- **Conversational success:** the model robustly understands valid English state descriptions, produces grammatical replies, and emits a legal action token that agrees with its English answer.
- **Planning success:** the model improves greedy reliability, reduces recovery-search work, or adds verified shorter paths when ensembled with the existing compact solver.

A polished conversational model can succeed even if it does not beat the compact solver. Solver improvement must be demonstrated separately with replay-verified path metrics.

---

## 2. What already exists

The LLM project starts from a much stronger foundation than a blank puzzle implementation.

### 2.1 Exact state and action engine

The existing Jewel implementation represents a state as:

- 12 permuted edge pieces;
- 12 edge orientations;
- six ring orientations in `C4`.

The reachable group has

```text
12! * 2^22 = 2,009,078,326,886,400
```

states. Every state has an exact 51-bit rank. The adapter in `official.py` proves the isomorphism between the compact coordinates and the competition's official 48-position representation for all twelve actions.

### 2.2 Exact and admissible supervision

The current assets include:

- an exact reverse ball through distance 8 containing 72,491,653 states;
- complete set-valued optimal-action masks for sampled exact states;
- two complete five-edge PDBs with 3,041,280 abstract states each;
- an admissible PDB lower bound;
- parity-aware IDA* with exact-ball completion;
- replay-verified public solution trajectories.

The whole graph cannot be traversed with BFS. The exact depth-8 ball is large but covers only a tiny fraction of the approximately `2.0e15` reachable states. Deep supervision must therefore remain a mixture of lower bounds, upper bounds, verified trajectories, search-generated labels, and on-policy corrections.

### 2.3 Current neural teacher and baseline

The existing structured model has:

- 4,803,621 parameters;
- six Transformer encoder blocks;
- width 256, eight attention heads, FFN width 1024;
- edge, orientation, ring, action-query, and global tokens;
- set-valued policy, geodesic, action-regret, scalar-distance, and distance-CDF heads.

Its held-out exact metrics are:

| Metric | Current result |
|---|---:|
| Top-1 descending action | 93.92% |
| Top-2 contains a descending action | 98.00% |
| Probability mass on descending actions | 0.919 |
| Exact distance MAE | 0.251 |

Its interactive search solves and officially verifies all 1,000 competition states. The raw score is 17,012; safely merging with the public incumbent gives 16,490.

These results define the baseline the language model must match or improve. Offline token loss is not an adequate comparison.

---

## 3. Critical problem-definition work

This section must be completed before any expensive training.

### 3.1 Decide what a colour observation means

The competition state contains 48 distinguishable labels. A human describes repeated colours. Those representations are not automatically equivalent.

The first implementation task is a `JewelColorCodec` that maps:

```text
official labelled state <-> physical colour observation <-> structured JewelState
```

There are two possible outcomes:

1. **Colours identify every piece and orientation.** For example, edge colour pairs may be unique. The codec can reconstruct one compact state exactly.
2. **Several labelled states produce the same colour observation.** The conversational task is then a quotient/Schreier-graph problem. The goal is any labelled state consistent with the solved colour pattern, not necessarily the identity-labelled state.

Do not assume the first case. Prove it with exhaustive piece-level checks and at least one million random round trips. If the observation is ambiguous, canonicalize the equivalence class or make search goal testing colour-based.

### 3.2 Freeze an English action lexicon

Create `action_lexicon.json` with exactly twelve machine actions and their human phrases:

```json
{
  "UBBBLL": {
    "face": 2,
    "direction": "right",
    "canonical_reply": "Turn face 2 to the right."
  },
  "-UBBBLL": {
    "face": 2,
    "direction": "left",
    "canonical_reply": "Turn face 2 to the left."
  }
}
```

The example mapping above is illustrative; the final mapping must be derived from the physical puzzle and verified against `puzzle_info.json`. Every phrase must map bijectively to one official generator.

### 3.3 Define the conversational contract

The production request should support both free English and a normalized form. A normalized training prompt could be:

```text
You are solving Christopher's Jewel. The goal is the solved colour state.

Current observation:
Side 1, clockwise from the marked corner: blue, blue, yellow, red, ...
Side 2, clockwise from the marked corner: ...
...

Legal commands:
- Turn face 1 to the left.
- Turn face 1 to the right.
...

Return exactly one move.
```

The assistant's machine-auditable response format should be:

```text
<MOVE_UBBBLL>
Turn face 2 to the right.
```

The special move token is authoritative. The sentence is the conversational realization. At inference, generation of the first token is constrained to the twelve move tokens or `<SOLVED>`.

### 3.4 Define correctness levels

The API and evaluation reports must distinguish:

- **legal:** the output names one valid generator;
- **verified-path:** the move is the first action of a complete path that replays to solved;
- **exact descending:** exact data prove that the move reduces shortest distance by one;
- **optimal:** an exact proof establishes that no shorter solution exists.

Outside the depth-8 ball, the model must never call a move optimal merely because its probability is high.

---

## 4. Target system architecture

```mermaid
flowchart LR
    A["User English state description"] --> B["Deterministic parser and colour codec"]
    B --> C["Validated symbolic Jewel state"]
    C --> D["Canonical English serializer"]
    D --> E["Causal Jewel language model"]
    E --> F["12-way move distribution"]
    E --> G["Regret, CDF, value, confidence heads"]
    F --> H{"Calibrated confidence gate"}
    G --> H
    H -->|"high confidence"| I["Greedy move"]
    H -->|"low confidence"| J["PDB-guided adaptive search"]
    J --> K["Depth-8 exact completion"]
    I --> L["Exact transition engine"]
    K --> L
    L --> M["Replay verifier"]
    M --> N["Move token plus English reply"]
```

### 4.1 Causal language backbone

Use a pretrained decoder-only model rather than enlarging the current encoder and merely calling it an LLM.

Primary configuration:

| Component | Choice |
|---|---|
| Base | Qwen3-0.6B-Base |
| Precision | BF16 |
| Maximum training length | 512 tokens initially; 1,024 only if needed |
| New tokens | structural tags, six colours, 48 positions if useful, 12 actions, solved/error tokens |
| Fine-tuning | LoRA for ablations; full-model tuning for finalists |
| Attention implementation | PyTorch SDPA or the model's supported fused path |
| Decoding | constrained first action token; temperature 0 for deployment |

The base model is intentionally small. A 0.6B model is large enough to preserve genuine English behavior but small enough for full fine-tuning and rapid iteration on one A100. Scale to 1.7B only after the 0.6B model passes the representation and objective gates.

### 4.2 Input tokenization

Train on three representations, always with the same underlying state and action labels:

1. **Compact symbolic text** - position tags plus colour tokens.
2. **Canonical English** - full side-by-side colour descriptions.
3. **Paraphrased English** - controlled variations of ordering, punctuation, colour wording, and request style.

The compact form tests planning. The English forms test language understanding. Mixing them makes it possible to diagnose whether a failure came from parsing or planning.

Example compact segment:

```text
<STATE> <P00> <BLUE> <P01> <BLUE> <P02> <YELLOW> ... </STATE>
<GOAL> <SOLVED_COLOURS> </GOAL>
<TASK_NEXT_MOVE>
```

### 4.3 Planner readout

At the final `<TASK_NEXT_MOVE>` position, expose:

- logits over the twelve move tokens;
- one scalar value estimate;
- a distance CDF;
- twelve action-regret predictions;
- one `needs_search` probability.

Tie the policy logits to the corresponding move-token embeddings if the implementation is stable; otherwise use a separate twelve-way head. The language-model vocabulary remains responsible for the subsequent English sentence.

### 4.4 Optional structured-teacher adapter

Do not begin with a complicated fusion model. First train from text alone.

If text-only training passes language gates but lags the 4.8M structured model, add an optional adapter:

```text
structured Jewel encoder -> projected state embedding -> causal LM prefix token
```

This creates a genuine hybrid LLM while allowing the compact expert to supply puzzle geometry. It must be evaluated against a simple wrapper control to determine whether the LLM is learning planning or merely verbalizing the teacher.

### 4.5 Optional plan decoder

Only after one-step policy training is strong, add one of these targets:

- two-to-four move macro;
- short subgoal state;
- compressed search trace;
- complete action sequence for shallow exact states.

Do not make long open-loop sequence generation the primary solver. Execute the first move exactly, observe the actual next state, and replan.

---

## 5. Dataset design

### 5.1 Unified record schema

Every stored example should retain machine labels even if the text is generated on the fly:

```json
{
  "state_rank": 1645971463294102,
  "official_state": [18, 19, 21, 20],
  "edge_perm": [9, 10, 1],
  "edge_ori": [0, 1, 0],
  "ring_ori": [0, 3, 1],
  "source": "exact|walk|public|search|dagger|dialogue",
  "exact_distance": null,
  "lower_bound": 8,
  "upper_bound": 15,
  "optimal_action_mask": null,
  "demonstrated_action": "-URBRBB",
  "sample_weight": 1.0,
  "split": "train",
  "template_family": "side_clockwise_v3",
  "episode_id": null
}
```

Text should usually be rendered online from this record. That prevents storing dozens of duplicated paraphrases and makes prompt-ablation experiments reproducible.

### 5.2 Source mixture

Start from the current 1.5M-state dataset, then expand it into the following logical pools:

| Pool | Initial proportion | Supervision |
|---|---:|---|
| Exact depth 1-8 | 40% | exact distance, complete optimal-action set, exact regret, CDF |
| Deep verified trajectories | 20% | demonstrated good action, path upper bound |
| Non-backtracking walks | 10% | demonstrated inverse action, walk upper bound |
| Exact transition/dialogue examples | 20% | next state, inverse action, language realization |
| DAgger/hard-state replay | 10% initially | teacher action sets, counterfactual rankings, confidence labels |

The hard-state fraction should grow to 30-50% during the final DAgger rounds, while retaining at least 25% exact anchor data.

### 5.3 Exact examples

For exact states, store every distance-decreasing move:

```text
A*(s) = {a : d(T(s,a)) = d(s)-1}
```

Do not choose one arbitrary reference move. During language realization training, sample one move from `A*(s)` to condition the English sentence, but mask that sampled move token from ordinary one-hot LM loss. Action selection is supervised by the complete set-valued loss.

### 5.4 Deep bounded examples

For deep states, combine:

- PDB lower bound `l(s)`;
- best verified path upper bound `u(s)`;
- demonstrated next action;
- all search-observed counterfactual actions;
- replay-graph improvements.

Do not pretend that `u(s)` is exact. Use interval supervision:

- CDF bins below `l(s)` are false;
- CDF bins at or above `u(s)` are true;
- bins in between are masked.

Unknown actions are unknown, not negative. A demonstrated action receives positive policy weight, but alternatives should become negative only when exact or counterfactual evidence supports that conclusion.

### 5.5 Public competition trajectories

Maintain two evaluation tracks:

- **strict generalization:** competition states and every state on their public paths are excluded from training;
- **transductive deployment:** public verified paths may be included, matching the current v4 model.

Never report transductive performance as held-out generalization. Current public-path fine-tuning contains only 8,529 distinct deep states despite 400,000 sampled records, so duplicate-aware splits remain essential.

### 5.6 Language variation

Use deterministic template families rather than uncontrolled paraphrase generation initially. Vary:

- “turn,” “rotate,” and “move” phrasing;
- left/right versus clockwise/counter-clockwise, only where the viewpoint is explicit;
- side numbering wording;
- commas, sentences, tables, and compact lists;
- colour spelling variants such as `color/colour` only when desired;
- whether the goal and legal-action list are restated;
- one-turn and multi-turn dialogue formats;
- harmless conversational text before the state.

Hold out entire template families for evaluation. Randomly splitting rendered examples would overestimate language generalization.

### 5.7 Multi-turn dialogues

Generate episodes of the form:

```text
User: Here is my current colour observation: ...
Assistant: <MOVE_X> Turn face 3 to the left.
Tool: The exact engine confirms the resulting symbolic state is ...
User: I made that turn. Now I see: ...
Assistant: <MOVE_Y> Turn face 1 to the right.
```

For production, the new observation is authoritative; dialogue history is context, not state truth. If the new observation contradicts the predicted transition, the model should ask the user to recheck the colours rather than silently continue from its imagined state.

---

## 6. Training objectives

### 6.1 Set-valued action loss

For exact states:

```text
L_set(s) = -log sum_{a in A*(s)} pi(a|s)
```

This is the primary planning loss. It does not punish an alternative optimal first move.

### 6.2 Demonstration policy loss

For non-exact verified trajectories:

```text
L_demo(s) = -log pi(a_demo|s)
```

Use a lower sample weight, initially `0.25`, unless the trajectory is the best verified path in a replay graph and has strong counterfactual evidence.

### 6.3 Exact regret loss

For exact states:

```text
Q*(s,a) = 1 + d(T(s,a))
r*(s,a) = Q*(s,a) - min_b Q*(s,b)
L_regret = SmoothL1(r_hat, r*)
```

Predicting centered regret is more directly useful than predicting twelve large absolute Q values. It answers how costly a local action is relative to the best action at that state.

### 6.4 Distance and CDF losses

Use exact Huber regression where distance is known and masked interval-CDF binary cross-entropy where only bounds are known.

The CDF is more useful than one scalar for confidence gating. It should expose when several remaining-distance ranges are plausible.

### 6.5 Language realization loss

Use completion-only causal LM loss on assistant text. The target action token is supplied by the policy branch; the language loss teaches the model to realize it as grammatical English.

For exact states with multiple optimal actions:

1. sample an optimal action;
2. teacher-force its move token without one-hot action loss;
3. train the following English phrase;
4. use `L_set` for the actual action decision.

This prevents natural-language SFT from undoing set-valued policy supervision.

### 6.6 Transition and inverse objectives

Given exact `(s,a,s')` triples, train auxiliary predictions of:

- the symbolic next state;
- the action given `(s,s')`;
- cycle consistency under `a` followed by `a^-1`.

These losses teach generator geometry but never replace exact transition execution.

### 6.7 Search-needed target

Define a binary target from calibrated evidence:

- `0` when the chosen action is exact-descending or a high-confidence greedy rollout succeeds within the quality gate;
- `1` for wrong top actions, low-margin failures, loops, failed rollouts, or cases where adaptive search improves the result.

This head should be calibrated after training, not trusted directly from raw logits.

### 6.8 Initial staged loss weights

Do not activate every term on day one. Use this staged starting point:

| Stage | Loss mixture |
|---|---|
| Format/transition pretraining | `1.0 L_LM + 0.5 L_transition + 0.2 L_inverse` |
| Exact policy | `1.0 L_set + 0.5 L_regret + 0.25 L_CDF + 0.10 L_value + 0.20 L_LM` |
| Mixed deep training | exact mixture above plus `0.25 L_demo + 0.20 L_interval-CDF` on bounded samples |
| DAgger | `1.0 L_set/listwise + 0.5 L_regret + 0.25 L_needs-search + 0.20 L_LM` |

Run one-component-at-a-time ablations before adopting the full mixture. Loss weights are engineering starting points, not scientific conclusions.

---

## 7. Training curriculum

### Phase 0: tokenizer and format smoke test

**Budget:** 5-10M tokens.

Goals:

- add and initialize special tokens;
- prove that all twelve actions can be emitted exactly;
- achieve 100% output parsing on a tiny overfit set;
- validate masking for set-valued action loss;
- validate all auxiliary heads and checkpoint reloads.

No larger run starts until the model can overfit 10,000 records and reproduce them after save/load.

### Phase 1: puzzle-language and transition adaptation

**Budget:** 100-150M tokens.

Train on:

- state descriptions;
- action lexicon examples;
- exact transitions;
- inverse transitions;
- solved-state recognition;
- invalid-input detection;
- short multi-turn dialogues.

The purpose is to align language tokens with puzzle structure before asking the model to make difficult planning decisions.

### Phase 2: exact policy and distance training

**Budget:** 250-400M tokens.

Train primarily on depth-1-8 exact states using:

- set-valued policy;
- exact regret;
- exact distance and CDF;
- language realization;
- 50% distance-balanced and 50% layer-size-weighted sampling.

Checkpoint selection is based on exact top-1/set mass, greedy rollouts, and calibration, not total training loss.

### Phase 3: deep mixed supervision

**Budget:** 200-300M tokens.

Add:

- public and search-verified trajectories;
- non-backtracking demonstrations;
- PDB lower/path upper interval labels;
- replay-graph action comparisons;
- strict-held-out competition-like random states.

Exact states remain at least 30% of every batch to prevent the distance scale and action geometry from drifting.

### Phase 4: DAgger and hard-state refinement

**Budget:** three rounds of 50-100M tokens each.

For each round:

1. Run greedy and confidence-gated rollouts.
2. Collect the first wrong action, first low-margin action, loops, repeated states, search disagreements, and unusually long paths.
3. Label with exact data where possible; otherwise use strong verified search, replay graphs, and bounded counterfactuals.
4. Retrain on `50% hard + 25% exact-uniform + 25% exact-distance-balanced`.
5. Refit confidence calibration on a separate validation split.

Puzzle 70 from the interactive demonstration should be a permanent hard-case canary. A successful model must either choose a verified better branch or correctly mark the raw decision as requiring search; it must not confidently present `UBBBLL` as a reliable decision.

### Phase 5: optional macro or search-trace training

Proceed only if one-step gates are passed.

Compare:

- direct two-to-four action macro;
- future subgoal state;
- compressed search trace;
- ordinary one-step replanning.

Every generated sequence is replayed through the exact engine. The first action is still the normal interactive response.

---

## 8. Optimizer and systems configuration

### 8.1 Full fine-tuning configuration

Initial 0.6B full-tuning settings:

| Setting | Initial value |
|---|---|
| Precision | BF16 |
| Sequence length | 512 |
| Global token batch | 65k-131k tokens via gradient accumulation |
| Learning rate | sweep `1e-5`, `2e-5`, `3e-5` |
| Weight decay | 0.1 |
| Warm-up | 2-3% of tokens |
| Schedule | cosine to 10% of peak LR |
| Gradient clipping | 1.0 |
| Optimizer | fused AdamW where supported |
| Checkpoint interval | every 1,000 optimizer steps and every evaluation gate |
| Evaluation interval | every 500-1,000 steps |

Use actual measured tokens/second from a 1,000-step smoke run to translate token budgets into wall time. Do not plan by nominal examples because English template lengths differ.

### 8.2 LoRA ablations

Use LoRA for rapid comparisons:

- ranks 16 and 32 initially;
- attention and MLP linear projections;
- learning rates `1e-4` and `2e-4`;
- train new embeddings and all puzzle-specific heads fully;
- keep the same data order and evaluation seeds as the full-tune control.

LoRA is an experiment accelerator, not automatically the final model. If full tuning materially improves exact action geometry, retain the fully tuned checkpoint.

### 8.3 Training implementation

Use Hugging Face Transformers plus TRL or a small custom Trainer. TRL supports conversational and prompt-completion datasets, completion-only loss, assistant-only loss, packing, and PEFT integration. A custom loss function is still required for set-valued policy, masked CDF, and auxiliary heads.

Project rules still apply:

- use `.venv/Scripts/python.exe`;
- BF16 and compile are acceptable for fixed-shape training;
- do not enable naive `torch.compile` for variable-shape search inference;
- save the tokenizer, model config, action lexicon hash, dataset manifest, and code revision with every checkpoint.

---

## 9. Four-week execution schedule

### Week 1: semantics, data, and controls

| Day | Work | Exit artifact |
|---:|---|---|
| 1 | Freeze colour observation, side ordering, goal, and action English | signed-off problem specification |
| 2 | Implement and test `JewelColorCodec` | million-state round-trip report |
| 3 | Implement action lexicon, parser, serializer, and constrained output parser | 12-action bijection tests |
| 4 | Build unified dataset manifest and duplicate-safe splits | dataset metadata and leakage audit |
| 5 | Generate canonical and held-out paraphrase families | prompt-template registry |
| 6 | Build deterministic English wrapper over the existing structured solver | non-LLM conversational control E0 |
| 7 | Run tokenizer/10k-record overfit smoke | Phase-0 gate report |

The A100 is used only for smoke tests this week. Most risk is semantic and data-engineering risk.

### Week 2: first genuine language model

| Day | Work | Exit artifact |
|---:|---|---|
| 8-9 | Qwen3-0.6B LoRA format/transition pretraining | E1 checkpoint |
| 10 | Evaluate parsing, English robustness, transitions, and legal output | E1 report |
| 11-12 | Train exact set-valued policy with auxiliary heads | E2 checkpoints |
| 13 | Loss ablations: set-only, +regret, +CDF | matched comparison |
| 14 | Select one configuration for full fine-tuning | Week-2 gate decision |

Do not scale model size in Week 2. The purpose is to prove that the representation, loss masks, and language interface work.

### Week 3: deep planning and on-policy correction

| Day | Work | Exit artifact |
|---:|---|---|
| 15-16 | Full fine-tune 0.6B on exact + bounded deep mixture | E3 checkpoint |
| 17 | Strict and transductive search evaluation | E3 report |
| 18 | Greedy/adaptive rollout collection | hard-state buffer v1 |
| 19 | DAgger round 1 | E4 checkpoint |
| 20 | DAgger round 2 plus calibration | E5 checkpoint and calibrator |
| 21 | Compare against 4.8M model at equal solve quality and neural-forward budget | Week-3 decision |

If DAgger does not reduce the first-error tail, stop adding data and diagnose representation/objective errors.

### Week 4: scale, sequence planning, and deployment

| Day | Work | Exit artifact |
|---:|---|---|
| 22-23 | Conditional 1.7B or structured-adapter experiment | E6 checkpoint |
| 24 | Conditional two-to-four move macro/search-trace experiment | E7 checkpoint |
| 25 | Integrate confidence-gated recovery with exact verification | interactive API |
| 26 | Run full 1,000-state competition evaluation and strict random benchmark | final metrics bundle |
| 27 | Distill/ensemble useful LLM decisions into compact solver; measure unique wins | contribution report |
| 28 | Freeze checkpoint, manifests, demo, model card, and reproducibility guide | release candidate |

The 1.7B and trace branches are conditional. If the 0.6B model fails fundamental planning gates, scaling is not the remedy.

---

## 10. Experiment matrix

All comparisons use identical state splits, prompt families, seeds where practical, and replay verification.

| ID | Model | Main change | Question |
|---|---|---|---|
| E0 | Existing 4.8M model + deterministic English renderer | no language model | How much of the user experience needs no LLM? |
| E1 | Qwen3-0.6B LoRA | ordinary one-hot action SFT | Can a pretrained LM learn the interface cheaply? |
| E2 | Qwen3-0.6B LoRA/full | set-valued action loss | Does correct multi-optimal supervision improve greedy rollouts? |
| E3 | E2 + regret/CDF/value | structured auxiliary heads | Do auxiliary targets improve calibration and recovery? |
| E4 | E3 + transition/inverse pretraining | generator geometry | Does dynamics pretraining improve sample efficiency? |
| E5 | E3/E4 + bounded deep mixture | lower/upper interval labels | Can it generalize beyond depth 8 without corrupting exact anchors? |
| E6 | E5 + DAgger | on-policy failures | Can hard-state correction remove low-margin tail errors? |
| E7 | Qwen3-1.7B | scale only | Is 0.6B capacity actually limiting? |
| E8 | 0.6B + structured prefix adapter | hybrid representation | Does the compact teacher add geometry beyond text? |
| E9 | 0.6B + macro/search trace | autoregressive planning | Does explicit short planning reduce recovery search? |
| E10 | Scratch 50-100M decoder | no language pretraining | Is general language pretraining useful beyond surface form? |

### Required ablations

Run these before drawing architectural conclusions:

- compact symbolic text versus canonical English;
- canonical English versus held-out paraphrase families;
- one-hot action CE versus set-valued loss;
- policy only versus policy + regret;
- scalar value versus CDF;
- exact-only versus exact + bounded deep data;
- offline training versus DAgger;
- LoRA versus full fine-tuning;
- raw greedy versus confidence-gated search;
- 0.6B versus 1.7B only if smaller-model gates pass.

The most important control is E0. If E0 and the LLM behave identically on all planner tests, the LLM is a conversational renderer, not a new solver.

---

## 11. Evaluation and acceptance gates

### 11.1 State and language correctness

| Metric | Minimum | Strong target |
|---|---:|---:|
| Valid colour descriptions parsed correctly | 99.99% | 100% |
| Invalid/ambiguous descriptions rejected | 99.9% | 99.99% |
| Move token is one of 12 legal actions | 100% | 100% |
| English sentence agrees with move token | 99.99% | 100% |
| Learned transition whole-state accuracy | diagnostic only | 99.999% |
| Exact-engine transition accuracy | 100% | 100% |

The learned transition head never defines production state, regardless of its score.

### 11.2 Exact planning metrics

Report by exact distance, not only as one aggregate:

- top-1 action belongs to `A*(s)`;
- top-2/top-3 contains an optimal action;
- total probability mass on `A*(s)`;
- expected exact action regret;
- distance MAE;
- CDF Brier score and log loss;
- symmetry consistency;
- greedy first non-geodesic decision depth.

Minimum final gate:

| Metric | Current compact model | LLM minimum | Stretch |
|---|---:|---:|---:|
| Exact top-1 | 93.92% | 95% | 99%+ |
| Exact top-2 | 98.00% | 98.5% | 99.9% |
| Optimal-set mass | 0.919 | 0.94 | 0.99 |
| Exact distance MAE | 0.251 | 0.30 or better | 0.15 |

The exact top-1 target should ultimately exceed 99% if greedy multi-step behavior is expected to be reliable. A 95% one-step policy is not a 95% solver.

### 11.3 End-to-end planning metrics

Measure:

- greedy solve rate;
- greedy loop/repeated-state rate;
- mean verified path length;
- path stretch against exact distance where known;
- solve rate and length at widths `1, 2, 4, 8, 16, 64`;
- parent neural forwards;
- generated transitions and unique states;
- wall time and peak memory;
- fraction handled greedily at calibrated risk thresholds;
- full replay-verification rate;
- unique per-puzzle wins after merging with the compact model.

Minimum deployment gate on the 1,000 competition states:

- 1,000/1,000 returned paths replay successfully in compact coordinates;
- 1,000/1,000 replay successfully in the official 48-position representation;
- no invalid action token;
- search-backed raw score no worse than the current 17,012 baseline at the matched search budget;
- all worse candidates are safely rejected by incumbent merging.

Stretch solver goal:

- at least one replay-verified per-instance improvement over the 16,490 incumbent, or a material reduction in neural-forward/search cost at equal path quality.

### 11.4 Calibration gate

Fit a small held-out calibrator using:

- top probability;
- top-two margin;
- policy entropy;
- predicted regret gap;
- CDF entropy;
- disagreement with the compact structured model;
- PDB lower-bound consistency.

Produce selective-risk curves. A candidate operating point is:

```text
empirical wrong-action rate < 1e-3 on states handled greedily
```

Use stricter thresholds only if the validation sample is large enough to estimate them honestly.

### 11.5 Strict reproducibility

Every result bundle includes:

- base-model identifier and revision;
- tokenizer hash;
- action-lexicon hash;
- dataset manifest and source counts;
- state-rank split function;
- template-family split;
- code revision;
- seed;
- training token count;
- exact checkpoint used;
- evaluation command;
- replay-verification output.

---

## 12. Interactive inference design

### 12.1 Three explicit modes

The user-facing API should expose:

1. **Raw model** - returns the causal LM's move distribution and uncertainty.
2. **Adaptive** - greedily executes only calibrated decisions and invokes small discrepancy search otherwise.
3. **Verified** - finds an entire solving path, independently replays it, and returns its first action.

Never label raw mode as verified.

### 12.2 Confidence-gated recovery

At a low-confidence state:

1. preserve the top two or three actions;
2. apply them with the exact engine;
3. batch short greedy rollouts;
4. rank using cumulative policy regret, CDF/value, and PDB lower bound;
5. globally deduplicate exact 51-bit state ranks;
6. terminate on an exact-ball hit;
7. replay the complete path before returning its first move.

This is preferable to inserting a 0.6B language model into a huge conventional beam. The language model is roughly two orders of magnitude larger than the compact expert and should be used at interactive decisions, ambiguous frontier states, or as an offline teacher—not blindly at millions of beam nodes.

### 12.3 Natural-language response policy

The normal response should be short:

```text
Turn face 2 to the right.
```

Optional structured metadata can include:

```json
{
  "move": "UBBBLL",
  "source": "model|adaptive_search|exact_ball",
  "confidence": 0.997,
  "verified_path": true,
  "remaining_path_length": 14
}
```

Do not generate unsupported claims such as “this is optimal” unless exact search proved them.

---

## 13. How the LLM can improve the compact models

The language model is too expensive to replace the compact scorer inside very wide search. Its most promising solver roles are:

### 13.1 Hard-state teacher

Run the LLM on states where compact models disagree or have low margins. Distill useful action distributions and uncertainty targets into the 4.8M Transformer or a fast Q/MLP model.

### 13.2 Root and discrepancy reranker

Use the LLM only at the initial state and a small number of ambiguous decisions. Preserve its alternative branch ordering without paying its cost at every frontier node.

### 13.3 Ensemble diversity

Merge verified paths from:

- compact Transformer;
- LLM greedy/adaptive solver;
- inverse-state/NISS frame;
- exact/PDB search;
- public incumbent.

The only meaningful evidence of diversity is unique replay-verified path improvements, not a different probability distribution.

### 13.4 Search-trace distillation

If the LLM learns compressed search traces or macros, distill those decisions into a small policy or macro head. This converts expensive offline reasoning into cheap inference.

---

## 14. Major risks and mitigations

| Risk | Why it matters | Mitigation |
|---|---|---|
| Colour descriptions lose labelled-state information | the conversational observation may be a quotient state | prove codec invertibility; otherwise solve the correct equivalence class |
| Model learns templates, not puzzle structure | synthetic English is repetitive | hold out template families and use compact-vs-English controls |
| One reference path penalizes alternative optima | creates false negatives | set-valued exact loss; mask arbitrary move token in LM loss |
| Deep path length treated as exact | corrupts value and regret | lower/upper interval supervision and unknown-action masks |
| Public-path leakage | inflates apparent generalization | strict and transductive tracks |
| Auxiliary losses interfere | combined loss can regress silently | staged training and one-term ablations |
| Greedy errors compound | one-step accuracy overstates solve rate | DAgger, first-error analysis, calibrated recovery |
| LLM is too slow for beam search | 0.6B is far larger than 4.8M | adaptive use, batching, distillation, compact ensemble |
| English reply disagrees with action | unsafe human interface | authoritative special move token plus deterministic verifier/renderer |
| Model hallucinates transitions | invalid internal state | exact engine always applies actions |
| Larger model hides objective flaws | expensive but uninformative scaling | 1.7B run only after 0.6B gates pass |
| “LLM” is only cosmetic | no new planner capability | E0 wrapper control and solver-contribution metrics |

---

## 15. Repository implementation plan

Add the new work beside the existing Jewel solver:

```text
jewel/
  llm/
    color_codec.py
    action_lexicon.json
    prompts.py
    tokenizer.py
    dataset.py
    collator.py
    model.py
    losses.py
    calibration.py
    adaptive_controller.py
    chat.py
  configs/
    llm_qwen06_lora_smoke.yaml
    llm_qwen06_exact.yaml
    llm_qwen06_mixed.yaml
    llm_qwen06_dagger.yaml
    llm_qwen17_scale.yaml
  scripts/
    30_build_llm_manifest.py
    31_validate_color_codec.py
    32_train_llm.py
    33_collect_llm_dagger.py
    34_eval_llm_policy.py
    35_eval_llm_interactive.py
    36_chat_llm.py
  tests/
    test_color_codec.py
    test_action_lexicon.py
    test_llm_loss_masks.py
    test_constrained_decode.py
    test_llm_replay.py
```

Expected durable artifacts:

- colour-codec specification and proof tests;
- action lexicon;
- duplicate-safe dataset manifest;
- prompt-template registry;
- base-model revision pin;
- staged checkpoints;
- exact-policy and language robustness reports;
- calibration plots;
- search-scaling curves;
- hard-state replay buffer;
- final model card;
- interactive demo using the exact engine.

---

## 16. Stop conditions and decision rules

Stop or redirect when any of these occurs:

1. **Codec failure:** colours do not determine a usable state and the quotient problem is not implemented.
2. **Representation failure:** after exact training, compact symbolic prompts greatly outperform English and language robustness does not improve with more templates.
3. **Objective failure:** set-valued top-1 remains below the current 93.92% after matched training.
4. **Calibration failure:** the model cannot identify its own wrong-action tail better than simple policy margin.
5. **Compute failure:** LLM adaptive inference costs more than compact search at equal verified path quality with no unique wins.
6. **Scale failure:** 1.7B improves language style but not action quality or calibration.

Redirect options:

- retain the LLM as a conversational parser/renderer around the compact expert;
- distill it into the compact model;
- focus on macro/search-trace learning rather than primitive action prediction;
- use the structured model as a prefix adapter;
- abandon full-model beam use and keep only root-level reranking.

---

## 17. Final success criteria

At the end of four weeks, a successful release should demonstrate this interaction:

```text
User: I have Christopher's Jewel. Side one is blue, blue, yellow, ...

Assistant: Turn face 2 to the right.

User: I turned it. Now side one is ...

Assistant: Turn face 4 to the left.
```

Under the surface, every turn must satisfy:

- the input state was validated;
- the output was one of twelve legal actions;
- the English phrase agreed with the action token;
- the exact engine applied the action;
- low-confidence states used recovery search;
- complete paths were replay-verified before being described as verified;
- optimality was claimed only when exact search proved it.

The scientific result should state separately whether the LLM:

- merely supplied the desired conversational interface;
- matched the structured policy;
- reduced search work;
- added unique shorter verified paths;
- produced useful hard-state knowledge that improved a compact model.

That separation is the difference between a persuasive demo and a meaningful LLM-planning experiment.

---

## 18. References and source material

### Project analyses

- [Training an LLM-style model to reason and plan on Cayley graphs](<../Training an LLM-style model to reason and plan on Cayley graphs.pdf>)
- [Feasibility of an LLM-Style Interactive Solver for Cayley-Graph Puzzles](<../Feasibility of an LLM-Style Interactive Solver for Cayley-Graph Puzzles.pdf>)
- [Current Jewel implementation and results](README.md)

### Base model and training stack

- [Qwen3-0.6B-Base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [Hugging Face TRL SFTTrainer documentation](https://huggingface.co/docs/trl/en/sft_trainer)
- [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)

### Planning and learning background

- [DeepCubeA: Solving the Rubik's cube with deep reinforcement learning and search](https://www.nature.com/articles/s42256-019-0070-z)
- [Subgoal Search for Complex Reasoning Tasks](https://proceedings.neurips.cc/paper/2021/hash/05d8cccb5f47e5072f0a05b5f514941a-Abstract.html)
- [Searchformer: Beyond A*](https://arxiv.org/abs/2402.14083)
- [DAgger](https://arxiv.org/abs/1011.0686)
- [Policy-Guided Heuristic Search](https://arxiv.org/abs/2103.11505)
- [CayleyPy RL](https://arxiv.org/abs/2502.18663)

