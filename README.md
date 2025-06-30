# CDRmix

**CDRmix** is an open, streaming-capable Mixture-of-Experts (MoE) large language model architecture built on top of RWKV-style blocks. It is optimized for long-context reasoning and modular expert routing, and is being developed for efficient deployment across heterogeneous hardware platforms — including NVIDIA, AMD, and Tenstorrent.

---

## Project Goals

CDRmix exists to:

- Enable fast, low-memory, **long-context LLMs**
- Combine **MoE scaling** with **RWKV-style streaming**
- Support **training and inference across multiple hardware targets**
- Emphasize **open training** on permissively licensed data
- Build out a new base for completely open source, domain-focused model work

---

## Current Status

| Model     | Params | Status         | Training Target          | Inference Target         |
|-----------|--------|----------------|--------------------------|--------------------------|
| `cdrmix-1b` | ~1.0B  | 🔄 In Progress | Cloud GPUs (A100/V100)   | Future: Tenstorrent cards|
| `cdrmix-4b` | ~4.0B  | 🔜 Planned     | TBD                      | TBD                      |
| `cdrmix-8b` | ~8.0B  | 🔜 Planned     | TBD                      | TBD                      |

---

## Architecture Highlights

- **RWKV-based experts**: linear-time, streaming-compatible transformer blocks
- **Sparse Mixture of Experts (MoE)** routing: only top-k experts activated per token
- **Modular training pipeline**: YAML-driven config system for dataset, model, training parameters
- **Hardware abstraction**: clean backend design to support CUDA, ROCm, and Tenstorrent
- **Token state-based memory**: supports inference on low-VRAM edge devices

---

## Training Configuration

The current training effort targets the **1B parameter** version (`cdrmix-1b`), using the [Common Pile v0.1 (Filtered)](https://huggingface.co/datasets/common-pile/common-pile-v01-filtered) as the only dataset.  This dataset should be sufficient for the 1b and 4b models, and additional data will be curated and published by Cascadian Digital Research for the 8b model.

Training pipeline features:

- Modular data loader
- Streaming `.jsonl` reading
- Configurable tokenizer, sequence length, and batching
- Ready for curriculum extension and multi-dataset support

---

## Repository Structure

cdrmix/
├── config/              # YAML configs for model, dataset, training
├── scripts/             # Training and evaluation shell scripts
├── src/
│   ├── cdrmix/          # Core model architecture (RWKV blocks, MoE logic)
│   ├── backends/        # Hardware backend support (CUDA, ROCm, TT)
│   ├── data_loader.py   # Dataset streaming pipeline
│   ├── train.py         # Main training loop
│   ├── eval.py          # Evaluation entry point
│   └── export.py        # (Stub) GGUF / ONNX export utilities
├── LICENSE              # MIT License
└── README.md            # This file

---

## Roadmap

### 🔜 Near-term

- ✅ Train `cdrmix-1b` on filtered Common Pile
- ✅ Finalize modular data pipeline
- ⬜ Add support for model export (GGUF, ONNX)
- ⬜ Tokenizer integration (BPE or custom)
- ⬜ First round of eval + sample outputs

### Mid-term

- ⬜ Launch training of `cdrmix-4b`
- ⬜ Integrate code datasets for fine-tuning
- ⬜ Begin inference and tuning for Tenstorrent deployment
- ⬜ Token-bucket batching and curriculum learning

---

## Contributing

CDRmix is an **open-source, community-oriented project**. Contributions are welcome via GitLab issues and merge requests. Please open an issue if you're interested in contributing adapters, dataset preps, or training configs.

If this project is useful to you, we also appreciate any social media sharing (@CascadianDigit on X) and would love honest and open feedback to make it better, even if you aren't comfortable with contribution.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

## Maintained by

**Cascadian Digital Research**  
Focused on high-performance, open, and resilient AI system design.

<https://gitlab.com/cascadiandigitalresearch>
