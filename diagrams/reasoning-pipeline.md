flowchart TD
  %% ===================== DATA LAYER =====================
  subgraph DATA[Data Ingestion & Prep]
    direction TB
    D0[HuggingFace Streaming Loader\n(Common Pile v0.1)]
    D1[Format Normalizer\n(JSONL/Text/Doc → text)]
    D2[Quality Filter\n(lang-id, length, heuristics, PII scrub)]
    D3[Near-Dedup & Hashing\n(SimHash/MinHash, URL/host dedup)]
    D4[Tokenizer Build/Load\n(cdrmix-bpe or sentencepiece)]
    D5[Streaming Tokenization\n(pack to fixed seq len, doc breaks)]
    D6[Curriculum Mixer\n(mix ratios, temperature sampling)]
    D7[Shard Writer\n(webdataset/tar + index)]
  end

  %% ===================== PIPELINE CONTROL =====================
  subgraph CTRL[Orchestration & Config]
    direction TB
    C0[Configs (*.yaml)\ncore + reason variants]
    C1[Param Autosizers\n(core & reasoning)]
    C2[Repro Runner\n(uv/conda + seed control)]
    C3[Experiment Registry\n(W&B/MLflow + checkpoints)]
    C4[CI: Lint/Unit/E2E tests\n(dataset, sampler, schedulers)]
  end

  %% ===================== BACKBONE TRAIN =====================
  subgraph PRE[Stage A — Backbone Pretraining (LM)]
    direction TB
    A0[RWKX-V Builder\n(25% Tx, 75% RWKV)]
    A1[Layer Scheduler\n(top-of-x | interleave-x)]
    A2[MoE Router\n(8 experts, top-k=2, aux losses)]
    A3[Dist Strategy\n(Data/Tensor/Pipeline/Expert parallel)]
    A4[Trainer Loop\n(CE loss, bf16, grad clip, ckpt/EMA)]
    A5[Eval Suite\n(PPL, zero-shot tasks, MoE balance)]
  end

  %% ===================== MEMORY & REASONING =====================
  subgraph REASON[Stage B — Reasoning Enablement]
    direction TB
    B0[Plasticity Adapters\n(plastic-LoRA, episodic only)]
    B1[Hebbian Proposer\n(local corr objectives)]
    B2[Working-Memory Summarizer\n(d↔S projections, stride)]
    B3[Episodic Memory Index\n(FAISS; write-on-accept)]
    B4[RL-MemAgent Heads\n(policy/value over memory ops)]
    B5[Policy Loop (GRPO-ish)\n(rewards: task, latency, stability)]
    B6[Acceptance & Rollback\n(branch scoring, short-term state)]
  end

  %% ===================== SCALE-OUT DATA PIPELINES =====================
  subgraph SCALE[Scale-Out Data (40B/200B)]
    direction TB
    S0[Extended Corpora\n(code, math, docs, curated web, books)]
    S1[Domain Balancing\n(per-domain caps, freshness weighting)]
    S2[Safety/License Filter\n(license tags, tox/safety blocklists)]
    S3[Curriculum v2\n(stage-wise lengths, task packs)]
    S4[Massive Sharding\n(multi-node epochs, resume-safe)
]
  end

  %% ===================== VALIDATION & RELEASE =====================
  subgraph VAL[Validation, Safety & Release]
    direction TB
    V0[Held-Out Dev/Val Splits\n(Common Pile derived + tasks)]
    V1[Long-Context Tasks\n(retrieval QA, tool-use sims)]
    V2[Routing Metrics\n(expert entropy, capacity hit rate)]
    V3[Safety/Content Filters\n(eval-only; red-team suites)]
    V4[Checkpoint Registry\n(S3/Drive + manifest + fprint)]
    V5[Model Card + Logs\n(citation, data policy, eval table)]
  end

  %% ===================== EDGES =====================
  D0 --> D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7
  D7 --> PRE
  CTRL --> PRE
  PRE --> REASON
  REASON --> VAL
  SCALE --> D6
  CTRL --> REASON
  CTRL --> VAL
