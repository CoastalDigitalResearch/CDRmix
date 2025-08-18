flowchart TD
  %% ===== Ingest & IO =====
  A[Input Tokens / Modalities] --> B[Shared Embedding & PosEnc\n(RWKV time-mix + RoPE for Tx-only)]
  B --> C[Reasoning Controller]

  %% ===== Reasoning Controller =====
  subgraph RC[Hybrid Hebbian–RL Reasoning System]
    direction TB
    C --> C1[Hebbian Proposer\n(ephemeral plasticity adapters)]
    C1 --> C2[RL Evaluator = RL-MemAgent\n(long-context policy over memory ops)]
    C2 --> C3[Decision: accept/adapt/reject\n(branch scoring, rollback, GRPO-style selection)]
    C3 --> C4[Update Short-Term State\n(gates/routing/adapters)]
  end

  %% ===== Memory & Tools =====
  subgraph MEM[Memory / Tools]
    direction TB
    M1[Streaming Working Memory\n(token-level state, KV/hidden summaries)]
    M2[External Episodic Memory\n(summaries, retrieval keys, traces)]
    M3[Tool API Hooks\n(optional: retrieval, calculators, etc.)]
  end
  RC --- MEM
  C4 --> D[Layer Scheduler]

  %% ===== Backbone Scheduler =====
  subgraph SCHED[Layer Scheduler]
    direction TB
    D[Variant = top-of-x | interleave-x\n%Tx = 25, %RWKV = 75]
    D -->|emit layer type| E
  end

  %% ===== Backbone with MoE =====
  subgraph BB[RWKX-V Backbone (Sparse MoE)]
    direction TB
    E{{Per-Layer Router\n(top-k experts)}} -->|tokens| F1[RWKV Block\n(TimeMix, ChannelMix)\n+ MoE-FFN experts]
    E -->|tokens| F2[Transformer Block\n(MHSA, MLP)\n+ MoE-FFN experts]
    style F1 fill:#f7f7f7,stroke:#bbb
    style F2 fill:#f7f7f7,stroke:#bbb
    F1 --> G[Residual + Norm]
    F2 --> G
  end

  G --> H[Readout Heads\n(next-token, value, routing aux)]
  H --> O[Losses]

  %% ===== Training Signals =====
  subgraph LOSS[Training & Signals]
    direction TB
    O[CE / SFT Loss] 
    O2[RL Rewards\n(from MemAgent policies, task metrics)]
    O3[Hebbian Plasticity Loss\n(local correlation objectives)]
    O4[MoE Aux Losses\n(load balance, z-loss, capacity)]
  end
  H -.-> O
  RC -.control signals.-> BB
  MEM -.retrieval/summarize.-> RC
  O2 -.credit assign.-> RC
