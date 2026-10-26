## Layer 1: Reflex & Expert Runtime (System 1)

The Reflex layer provides fast, reactive responses at high frequency (10-100Hz) using specialized expert policies.

**Architecture**:
```
┌───────────────────────────────────────────────┐
│          Supervisor (Rust)                    │
│  - Real-time scheduling (10-100Hz)            │
│  - Safety checks and constraints              │
│  - Multi-expert coordination                  │
└──────────────────┬────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────┐
│       Expert Runtime (Python)                 │
│                                               │
│  Base Model (Shared)                         │
│  ├─ Vision Transformer (ACT/DiT/Diffusion)   │
│  └─ Shared parameters across experts         │
│                                               │
│  LoRA Adapters (Hot-Swappable)              │
│  ├─ reach_expert.safetensors                 │
│  ├─ grasp_expert.safetensors                 │
│  ├─ push_pull_expert.safetensors             │
│  ├─ cloth_expert.safetensors                 │
│  └─ ... (11 expert tiers)                    │
└───────────────────────────────────────────────┘
```

**Expert Tiers** (11 total):
- **Tier 1** (100 epochs): reach_expert, observe_expert
- **Tier 2** (150 epochs): grasp_expert, place_expert, lift_expert
- **Tier 3** (200 epochs): push_pull_expert, twist_expert, insert_expert
- **Tier 4** (300 epochs): cloth_expert, rope_expert
- **Tier 5** (400 epochs): bimanual_expert

**LoRA Hot-Swapping**:
- Load base model once (memory-efficient)
- Swap LoRA adapters at runtime (<100ms)
- No model reloading required
- Supports multi-GPU deployment

**Communication**: Hybrid Bus (ZeroMQ/Unix sockets) for low-latency IPC