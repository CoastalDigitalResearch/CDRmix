flowchart TD
  A[Input Tokens / Modalities] --> B[Shared Embedding & PosEnc\n(Token Emb + RoPE for T-blocks)]
  B --> S[Layer Scheduler\n(top-of-x | interleave-x)\n25% Transformer / 75% RWKV]

  subgraph BB[RWKX-V Backbone (Sparse MoE)]
    direction TB
    S --> R1[RWKV Block\n(TimeMix, ChannelMix)]
    S --> T1[Transformer Block\n(MHSA, MLP)]
    R1 --> N1[Residual + Norm]
    T1 --> N1

    %% MoE applies to FFN/ChannelMix paths
    N1 --> ROUTER{{MoE Router\n(top-k experts)}}
    ROUTER -->|dispatch| EXP1[Expert FFN 1]
    ROUTER -->|dispatch| EXP2[Expert FFN 2]
    ROUTER -->|...| EXPk[Expert FFN k]
    EXP1 --> MERGE[Merge & Residual]
    EXP2 --> MERGE
    EXPk --> MERGE
  end

  MERGE --> H[Readout / LM Head]
  H --> L[Training Losses\n(CE + MoE aux: load balance, capacity, z-loss)]
