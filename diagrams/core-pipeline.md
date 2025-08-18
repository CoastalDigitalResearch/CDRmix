flowchart TD
  %% ===================== DATA LAYER =====================
  subgraph DATA[Data Ingestion & Prep]
    direction TB
    D0[HuggingFace Streaming Loader\n(Common Pile v0.1)]
    D1[Format Normalizer\n(JSONL/Text/Doc → text)]
    D2[Quality Filter\n(lang-id, length, PII scrub, badlines)]
    D3[Near-Dedup & Hashing\n(SimHash/MinHash; URL/host dedup)]
    D4[Tokenizer Build/Load\n(cdrmix-bpe or sentencepiece)]
    D5[Streaming Tokenization\n(pack to fixed seq len; doc breaks)]
    D6[Curriculum Mixer\n(domain/length mix; temperature sampling)]
    D7[Shard Writer\n(webdataset/tar + index; resume-safe)]
  end

  %% ===================== ORCHESTRATION =====================
  subgraph CTRL[Orchestration & Repro]
    direction TB
    C0[Configs (*.yaml)\n(cdrmix-core sizes)]
    C1[Param Autosizer\n(core_param_autosizer.py)]
    C2[Run Harness\n(uv/conda; seeds; determinism)]
    C3[Experiment Registry\n(W&B/MLflow; ckpt manifest)]
    C4[CI: Lint/Unit/E2E\n(loaders, samplers, schedulers)]
  end

  %% ===================== TRAINING CORE =====================
  subgraph PRE[Stage A — Core LM Pretraining]
    direction TB
    A0[RWKX-V Builder\n(25% Transformer, 75% RWKV)]
    A1[Layer Scheduler\n(top-of-x | interleave-x)]
    A2[MoE Router\n(8 experts; top-k=2; aux losses)]
    A3[Distributed Strategy\n(DDP + TP/PP + Expert Parallel)]
    A4[Trainer Loop\n(CE loss; bf16; grad clip; EMA; ckpt)]
    A5[Eval Suite\n(PPL; zero-shot; MoE balance/capacity)]
  end

  %% ===================== SCALE-OUT FOR LARGE MODELS =====================
  subgraph SCALE[Scale-Out Data & Throughput (40B/200B)]
    direction TB
    S0[Extended Corpora\n(code, math, docs, curated web, books)]
    S1[Domain Balancing\n(per-domain caps; freshness weights)]
    S2[Safety/License Filter\n(license tags; tox/safety blocklists)]
    S3[Curriculum v2\n(stage-wise lengths; long-seq packs)]
    S4[Massive Sharding\n(multi-node epochs; fault tolerant)]
  end

  %% ===================== VALIDATION & RELEASE =====================
  subgraph VAL[Validation & Release]
    direction TB
    V0[Held-Out Splits\n(Common Pile derived + task sets)]
    V1[Routing Metrics\n(expert entropy; hit rate; overflow)]
    V2[Long-Seq Sanity\n(context growth; loss vs len)]
    V3[Checkpoint Registry\n(S3/Drive + manifest + fingerprint)]
    V4[Model Card & Specs\n(specs/architecture_backbone.md)]
  end

  %% ===================== EDGES =====================
  D0 --> D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7
  D7 --> PRE
  CTRL --> PRE
  SCALE --> D6
  PRE --> VAL
  CTRL --> VAL
