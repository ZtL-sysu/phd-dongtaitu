# phd-dongtaitu — Offline Event-Driven Spiking Dynamic Graph Networks (EDSGN)

> **中文简介**：本仓库是博士课题《*Event Driven Spiking Dynamic Graph Networks for Nonstationary Temporal Graph Forecasting*》的**离线工程实现与系统实验仓库**（RQ1–RQ3，涵盖稀疏动态图学习、关系条件化快慢脉冲记忆与冻结权重后的预算调度三大模块）。仓库包含 20 页博士研究计划原文、离线三模块详细实施方案、完整 Python 实现包 `edsgn`（含单元测试与阶段脚本）、P0–P3 全流程协议/配置/独立评审记录、以及截至 **2026-10-13** 已完成的 P3 阶段 **68 任务基线战役**（DGCRN / GraphWaveNet / STAEformer / STID / AGCRN / DCRNN / GRU / SeqSNN / V0 / V1）的 JSON/CSV 验证集结果与全量哈希审计回执。**模型权重（`.pt`）、预测数组（`.npz`）、原始数据集与第三方上游源码仓库保留在本地，不上传至 GitHub。**

Offline engineering implementation, preregistered experiment protocols, audit logs, and baseline campaign evidence for the doctoral research project **Event-Driven Spiking Dynamic Graph Networks for Nonstationary Temporal Graph Forecasting** (RQ1–RQ3).

- **Current Pipeline Status (as of 2026-10-13):**
  - `P0–P2`: **PASSED** (data contracts, semantic tests T01–T26 / 211 integrated checks, resource feasibility & AGCRN training-only memory repair).
  - `P3` (Baseline Campaign): **RUNNING** — 68/68 serial baseline jobs completed (34.3h wall time; 44/68 passed full independent hash + MAE recomputation audit; 24/68 DGCRN/SeqSNN multi-seed rows pending manifest-chain reconciliation correction; 4/4 legacy simple baseline train-only fit audits passed with `max_abs = 0.0`).
  - `P3A` (Candidate Config Gate), `P4` (Core V0–V9 Matrix — proposed spiking/retention variants V2–V9 not yet trained), `P5/P5A` (Frozen Test & Holm Statistical Inference): **PENDING**.
  - `P6+` (Manuscript drafting / submission): **Out of scope** for this offline experiment run.
- **Related Repositories:**
  - [`ZtL-sysu/phd-snn`](https://github.com/ZtL-sysu/phd-snn) — Parent PhD SNN & GNN→SNN research hub.
  - [`ZtL-sysu/gnn2snn-v2`](https://github.com/ZtL-sysu/gnn2snn-v2) — Journal-track training-free GNN→SNN conversion & reproducibility audit manuscripts.
  - [`ZtL-sysu/gnn2snn-dynamic-graph`](https://github.com/ZtL-sysu/gnn2snn-dynamic-graph) — Lightweight dynamic-graph GNN→SNN conversion collaboration repository.
  - [`ZtL-sysu/lite-sdgn-code`](https://github.com/ZtL-sysu/lite-sdgn-code) — Reference code package for the Lite-SDGN dynamic graph learning paper.

---

## 1. Research Overview & Three Core Modules

The doctoral research proposal ([`PhD_Research_Proposal.pdf`](PhD_Research_Proposal.pdf)) and offline engineering specification ([`博士RP_离线三模块详细实施方案.md`](博士RP_离线三模块详细实施方案.md)) target nonstationary multistep node-activity forecasting across three coordinated modules:

1. **Module 1 — Confidence-Aware Sparse Dynamic Graph (RQ1 / H1):**
   - Learns sparse causal predictive edges (at most $k=8$ incoming neighbors per node from a candidate pool $K=32$) that distinguish credible structural regime changes from observation noise and missing records.
2. **Module 2 — Relation-Conditioned Fast/Slow Spiking Memory (RQ2 / H2):**
   - Genuine Leaky Integrate-and-Fire (LIF) spiking memory ($d_{\text{total}}=64$, partitioned into 16 channels each for fast/slow $\times$ local/neighbor state), where graph change magnitude $c$ modulates neighbor-memory retention $g$ so stable background dynamics are preserved during abrupt neighbor switches.
3. **Module 3 — Frozen Budgeted Graph & State Scheduler (RQ3 / H3):**
   - Deterministic post-freeze scheduler with independent graph-refresh ($G$) and state-refresh ($S$) switches that genuinely skip graph/state recomputation under explicit compute budgets, benchmarked with full batch-1 wall-time and counter profiling.

### Model Variant Taxonomy (`V0–V9`)

| Family / Variant | Description | Role |
|---|---|---|
| `V0` | Static-graph continuous ANN | Internal baseline (B2) |
| `V1` | Confidence-aware dynamic-graph continuous ANN | Internal baseline (B2) |
| `V2` | Dynamic-graph + fast/slow continuous memory control | Internal ANN control / `V8` freeze source |
| `V3` | Dynamic-graph + LIF spiking memory without relation-conditioned retention | H2 ablation |
| `V4` | **Full EDSGN** (confidence-aware sparse dynamic graph + relation-conditioned fast/slow LIF memory) | Full proposed model (`V5/V6/V7` freeze source) |
| `V5 / V6 / V7` | Frozen `V4` checkpoint replayed under independent $G$/$S$ scheduling policies | H3 selective execution |
| `V8` | Frozen `V2` continuous checkpoint replayed under the matched scheduler | H3 ANN vs. SNN control |
| `V9` | `V4` with confidence gating disabled | H1 confidence ablation |

---

## 2. Datasets & Experimental Domains

| Domain | Type & Granularity | Role & Current Observation |
|---|---|---|
| `synthetic-n64-identity-change0.3-miss0` | Controlled 64-node nonstationary simulation (known ground-truth regime switches; plus stress suites: `identity10/30`, `weight10/30`, `gradual128`, `nonlinear`, `noiseonly`, `localshock`, `N256`) | Semantic verification & structural H1/H2 recovery evaluation. Primary default domain is near the noise floor (~0.0258 MAE); stress domains carry the primary structural discrimination signal. |
| `email` (`email-Eu-core-temporal`) | Daily node-activity forecasting (SNAP) | Primary real dataset (descriptive row: official validation split exhibits zero-target degeneration, documented in C02). |
| `email-episode` | Earliest active episode extracted from `email-Eu-core-temporal` | Additive diagnostic real domain with dense non-zero activity. |
| `mooc` (`act-MOOC`) | Hourly user-action activity forecasting (SNAP; dropout labels strictly excluded) | Primary non-degenerate real-world forecasting benchmark. |

---

## 3. Current Progress & P3 Baseline Validation Results

Full status assessment, audit details, and remaining budget analysis are recorded in [`f-research-v1-run-20261011-offline-edsgn/项目状态评估与推进方案-2026-10-13.md`](f-research-v1-run-20261011-offline-edsgn/项目状态评估与推进方案-2026-10-13.md).

### Pipeline Gate Ledger (`pipeline-state.md`)

| Gate | Status | Key Evidence |
|---|---|---|
| `P0` | PASSED | `protocols/p0-gate.md`, `path-containment-manifest.json` |
| `P1 / P1A` | PASSED | `protocols/guidance-extraction.md`, `mechanism-card.md`, `claim-evidence-matrix.md`, `strong-result-feasibility-lock.md` |
| `P1B` | PASSED | `reviews/ccfa-pre-experiment-review.md`, `reviews/p1b-revalidation-v2.md` |
| `P2` | PASSED | `protocols/p2-engineering-delta-gate-v4.md`, 211 integrated checks (`reviews/p2-c05-integrated-tests-v1.md`), AGCRN memory repair |
| `P3` | **RUNNING** | 68/68 baseline jobs completed; 44/68 full hash + MAE audit passed; 4/4 simple-baseline train-only fit audits passed (`max_abs=0.0`); 24 multi-seed rows pending C08 manifest-chain reconciliation; `V2` seed0 pending |
| `P3A / P4 / P5 / P5A` | PENDING | Proposed variants (`V2–V9`), 49 preregistered primary endpoints (H1/H2/H3), paired block bootstrap + Holm correction |

### P3 Validation MAE Summary (Weighted Pooled Horizons 1/3/7; Test Split Strictly Frozen)

| Method | `synthetic-n64` (s0 / 5-seed) | `email-episode` (s0 / 5-seed) | `mooc` (s0 / 5-seed) | `email` (zero-target val, descriptive only) |
|---|---:|---:|---:|---:|
| **GraphWaveNet** | 0.02577 | **0.61099** | 3.11789 | 0.01058 |
| **STAEformer** | 0.02578 | 0.61944 | **3.05861** | 0.01344 |
| **STID** | — | 0.65653 | 3.08229 | — |
| **DGCRN** | 0.02577 (×5) | 0.62644 | 3.60486 (3.44–3.73) | 0.00186 (0.003–0.012) |
| **DCRNN** | 0.02577 | 0.66731 | 4.18726 | 0.01875 |
| **AGCRN** | 0.02577 | 0.67102 | 3.73743 | 0.11994 |
| **GRU** | 0.02585 (×5) | 0.69054 (0.684–0.776) | 3.849–6.222 | 0.01951 (0.018–0.035) |
| **SeqSNN** | 0.02602 (×5) | 0.91059 | 7.87874 (7.878–7.883) | 0.41972 (0.002–0.572) |
| **V0 (Static ANN)** | 0.02583 | 0.75832 | 4.11879 | 0.03547 |
| **V1 (Dynamic ANN)** | 0.02582 | 0.72112 | 4.26916 | 0.04780 |

> **Honest Scientific Note:** As of 2026-10-13, **no proposed spiking/retention variant (`V2–V9`) has been trained yet** and **no test-split evaluation has been executed**. The table above establishes only that the baseline suite is solid and that `V0/V1` continuous ANN controls currently lag strong spatiotemporal baselines on `email-episode` and `mooc` (with `V1` outperforming `V0` on `email-episode`). Whether the full spiking/retention mechanism (`V4`) and frozen scheduler (`V7`) close the gap or achieve favorable accuracy–cost trade-offs remains to be tested in `P4–P5A`.

---

## 4. Repository Structure

```text
phd-dongtaitu/
├── README.md                                            # This overview
├── .gitignore                                           # Excludes checkpoints, datasets, third-party clones, caches
├── PhD_Research_Proposal.pdf                            # 20-page doctoral research proposal (anonymized)
├── 博士RP_离线三模块详细实施方案.md                         # Detailed Chinese engineering spec for Modules 1–3
└── f-research-v1-run-20261011-offline-edsgn/            # Canonical offline research run root
    ├── 项目状态评估与推进方案-2026-10-13.md                # Comprehensive status, audit & budget assessment
    ├── pipeline-state.md / pipeline-state.json          # Canonical P0–P21 stage gate ledger
    ├── execution-contract-v1.md                         # Offline execution & leak-prevention contract
    ├── experiment-traceability.md                       # Preregistered queue traceability matrix (D1–V1)
    ├── correction-ledger.md / correction-loop-state.md  # Versioned protocol corrections & reconciliation state
    ├── run_control.py & *.py                            # Queue approval, stage-gate & resource-freeze controllers
    ├── inputs/                                          # Proposal PDF, extracted text & implementation plan
    ├── edsgn/                                           # Installable Python package (edsgn-offline v0.1.0)
    │   ├── pyproject.toml & README.md                   # Package metadata & CLI usage guide
    │   ├── src/edsgn/                                   # Core modules: model, scheduling, calibrate, profile,
    │   │                                                #   data, prepare, engine, train, evaluate, robustness,
    │   │                                                #   statistics, baselines, rp_baselines, external_baselines,
    │   │                                                #   dcrnn_baseline, stid_baseline, agcrn_checkpoint, prior_cache
    │   ├── scripts/                                     # Experiment runners, auditors & frozen-pipeline validator
    │   ├── tests/                                       # Semantic & integration unit tests (T01–T26)
    │   └── staging/                                     # Versioned engineering staging scripts
    ├── configs/                                         # Training & campaign JSON configs (smoke, bound-v3, catalogs)
    ├── protocols/                                       # Frozen contracts, endpoint plans & replay-row definitions
    ├── reviews/                                         # Independent & root engineering/scientific review records
    ├── runs/                                            # Completed P3 baseline training summaries & logs (JSON/CSV only)
    ├── logs/                                            # Structured JSON audit receipts, phase wall ledgers & fit checks
    ├── snapshots/                                       # Immutable pre-delta Python/JSON snapshots (zip archives excluded)
    └── data/                                            # Dataset provenance manifests only (raw/processed binaries excluded)
```

---

## 5. Package Modules & Quick Start (`edsgn`)

### Environment

- Python `>=3.11` with `torch`, `numpy`, `scipy`, `pandas`, `pyyaml`, `matplotlib`, `psutil` (developed in local Conda environment `dyg-cold`; supports CUDA and Apple Silicon MPS/CPU).

### Core Modules (`f-research-v1-run-20261011-offline-edsgn/edsgn/src/edsgn/`)

| Module | Responsibility |
|---|---|
| `prepare.py`, `data.py`, `prior_cache.py` | Causal dataset preparation, train-only normalization/priors, origin splits (no test leakage) |
| `model.py` | Sparse dynamic graph learner, fast/slow local/neighbor LIF cells, relation-conditioned retention, `V0–V9` |
| `scheduling.py`, `calibrate.py` | Frozen post-training $G$/$S$ selective scheduling & validation-only budget calibration (`--epsilon`) |
| `engine.py`, `train.py`, `evaluate.py` | Reproducible training loop, checkpoint freezing (`test-freeze.json`), masked raw-unit evaluation |
| `profile.py` | Batch-1 float32 end-to-end wall-time, CUDA-synced timing, component diagnostics & operation counters |
| `robustness.py`, `statistics.py` | Missingness/noise perturbations, paired forecast-origin block bootstrap & Holm correction |
| `baselines.py`, `rp_baselines.py`, `external_baselines.py`, `dcrnn_baseline.py`, `stid_baseline.py`, `agcrn_checkpoint.py` | Simple baselines + pinned author-code adapters for DGCRN, GraphWaveNet, AGCRN, DCRNN, STID, STAEformer, SeqSNN |

### Running Tests & Commands

From `f-research-v1-run-20261011-offline-edsgn/edsgn/`:

```bash
export PYTHONPATH="$(pwd)/src"

# Run core semantic & integration unit tests
python -m pytest tests -m "not pending_training" -q

# Prepare datasets (synthetic generates locally; email/mooc require SNAP raw files in ../data/raw/)
python -m edsgn.prepare --dataset synthetic
python -m edsgn.prepare --dataset email
python -m edsgn.prepare --dataset mooc

# Train a configured model / baseline
python -m edsgn.train --config ../configs/synthetic-baseline.json

# Evaluate simple baselines on validation split
python -m edsgn.evaluate --simple-baselines \
  --dataset-path ../data/processed/email/email.npz --split val --output ../runs/baselines/email-simple

# Calibrate frozen scheduler on validation split (post-P4)
python -m edsgn.calibrate --checkpoint ../runs/V4/seed0/checkpoint.pt \
  --output ../runs/V4/seed0/calibration --queue ../protocols/pilot-queue.json --epsilon 0.02 --device cpu

# Measure batch-1 end-to-end latency & counters (requires frozen test-freeze.json)
python -m edsgn.profile --checkpoint ../runs/V4/seed0/checkpoint.pt \
  --dataset-path ../data/processed/synthetic-n64-identity-change0.3-miss0 \
  --split test --output ../profiles/V4/seed0/cpu --device cpu
```

---

## 6. External Baseline Sources (Excluded from Git, Pinned Here)

To keep the repository lightweight, the cloned third-party repositories under `f-research-v1-run-20261011-offline-edsgn/sources/` (~849 MB) are git-ignored. Restore them at the exact audited commits below when running external baseline adapters:

| Directory | Upstream Repository | Pinned Commit |
|---|---|---|
| `AGCRN` | `https://github.com/LeiBAI/AGCRN.git` | `7fbbf2a` |
| `DCRNN` | `https://github.com/liyaguang/DCRNN.git` | `602afd9` |
| `DCRNN_PyTorch` | `https://github.com/chnsh/DCRNN_PyTorch.git` | `d92490b` |
| `DGCRN` | `https://github.com/LakeChuan/DGCRN.git` | `b9f8e40` |
| `Graph-WaveNet` | `https://github.com/nnzhan/Graph-WaveNet.git` | `6b162e8` |
| `STAEformer-pinned` | `https://github.com/XDZhelheim/STAEformer.git` | `fc49d39` |
| `STID` | `https://github.com/GestaltCogTeam/STID.git` | `e8b313b` |
| `SeqSNN` | `https://github.com/microsoft/SeqSNN.git` | `15cbacb` |

---

## 7. Excluded Local-Only Artifacts

- **Trained model checkpoints & prediction arrays** (`*.pt`, `*.npz`, `*.npy`, ~1.15 GB in `runs/`) — validation summaries (`training_summary.json`, `training_log.csv`, etc.) are committed in `runs/`.
- **Raw & processed datasets** (`data/raw/`, `data/processed/**/*.npz`, ~269 MB) — provenance manifests (`manifest.json`, `data-manifest.json`) are committed in `data/`.
- **Third-party source clones** (`sources/`, ~849 MB) — pinned commits listed in Section 6.
- **Snapshot zip bundles, raw `.log` streams, hidden `.c06-*` verification caches, `.pytest_cache/`, `.workbuddy/`** — structured JSON audit receipts in `logs/` and Python/JSON snapshots in `snapshots/` are committed.
