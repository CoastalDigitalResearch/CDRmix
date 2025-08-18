flowchart LR
  subgraph RWKV_Block
    X0[Input] --> TM[TimeMix] --> CM[ChannelMix]
    CM --> RN[Residual + Norm]
  end

  subgraph Transformer_Block
    Y0[Input] --> ATTN[MHSA (RoPE)] --> MLP[FFN]
    MLP --> RN2[Residual + Norm]
  end

  %% MoE wraps FFN/ChannelMix
