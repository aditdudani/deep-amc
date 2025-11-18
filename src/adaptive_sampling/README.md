 # Adaptive Sampling Pipeline
 
 ## Overview
 Adaptive sampling dynamically reallocates training exposure across (class, SNR) buckets based on where the model is currently weaker. After each epoch (post warmup & accuracy gate) confusion statistics per SNR drive a weight update, increasing probability mass for harder buckets (higher error) and gently reducing emphasis on easier buckets.
 
 ## Key Concepts
 - Buckets: Cartesian product of classes × SNRs. With 8 classes and 6 SNRs ⇒ 48 buckets.
 - Weights: 2D matrix `weights[class][snr]` summing to 1. Used to sample training examples next epoch.
 - Warmup: Initial epochs use uniform weights to stabilize feature learning.
 - Gate: Require validation accuracy ≥ threshold (e.g. 0.15) before adaptive updates to avoid noise-driven shifts.
 - Update (simplified): Form error‐proportional target distribution (with epsilon floor & cap) then smooth: `w_{t+1} = (1−β) w_t + β target`; renormalize.
 - Guardrails: warmup epochs, min-val-acc gate, epsilon floor, max_cap per bucket, smoothing β.
 
 ## Modes
 | Mode | Description | Use Case |
 |------|-------------|----------|
 | parity | Baseline tf.data directory loader; no adaptation | Sanity check environment & architecture |
 | sampler-uniform | Sampler backends; uniform weights fixed | Parity while exercising sampler paths |
 | adaptive | Confusion-driven weight updates post warmup | Target robustness & tail improvement |
 
 ## Sampler Backends
 | Backend | Mechanism | Pros | Cons |
 |---------|-----------|------|------|
 | sequence | Keras Sequence; per-batch bucket draws | True dynamic sampling | Python overhead |
 | tfdata | Pre-sample epoch → tf.data decode | Fast, parallel IO | Needs per-epoch resample for adaptation |
 | tfdata-dir-class | Uniform class sampling directly from dirs | No metadata dependency | Ignores SNR buckets |
 
 Enable per-epoch resampling with `--resample-each-epoch` (critical for tfdata adaptation).
 
 ## Recommended Progression
 ```bash
 # 1. Parity baseline
 python src/adaptive_sampling/train_squeezenet_sampler.py --mode parity
 
 # 2. Uniform sampler parity
 python src/adaptive_sampling/train_squeezenet_sampler.py \
	 --mode sampler-uniform --sampler-backend tfdata --uniform-scope class \
	 --resample-each-epoch --valprobe-batches 4
 
 # 3. Adaptive (class scope first)
 python src/adaptive_sampling/train_squeezenet_sampler.py \
	 --mode adaptive --sampler-backend tfdata --uniform-scope class \
	 --warmup-epochs 3 --min-val-acc 0.15 --lr 0.005 --clipnorm 5.0 \
	 --resample-each-epoch --valprobe-batches 4
 
 # 4. Optional: full (class,SNR) scope later
 python src/adaptive_sampling/train_squeezenet_sampler.py ... --uniform-scope class_snr
 ```
 
 ## Important Flags
 | Flag | Purpose |
 |------|---------|
 | `--warmup-epochs` | Delay updates until base accuracy established |
 | `--min-val-acc` | Skip updates if validation accuracy below threshold |
 | `--uniform-scope` | Granularity for uniform start / adaptation |
 | `--resample-each-epoch` | Rebuild sampled tf.data each epoch |
 | `--clipnorm` | Gradient norm clipping for stability |
 | `--valprobe-batches` | Lightweight early slice probe |
 | `--debug-trainstep` | Spot-check loss/acc & weight delta |
 
 ## Diagnosing Issues
 | Symptom | Likely Cause | Fix |
 |---------|--------------|-----|
 | Chance from start | Label mapping or double normalization | Folder-based labels; single Rescaling |
 | Early gain then collapse | LR/momentum + static sampling | Lower LR; enable resampling; add clipnorm |
 | No weight changes | Warmup/gate blocking | Adjust thresholds; confirm val accuracy |
 | Over-concentration | Aggressive β / low ε | Lower β; raise ε; enforce cap |
 
 ## Weight Inspection & Metrics
 Use:
 ```bash
 python src/adaptive_sampling/inspect_weights.py \
	 --dir results/adaptive_sampling/<RUN_TAG> --top 12 --entropy
 ```
 Outputs top buckets, per-SNR share, entropy `H`, ratio `H/H_uniform`, and KL divergence `KL(p||u)=log(N)-H`.
 - Uniform entropy: `log(C*S)`.
 - Heuristic: `H/H_uniform` < 0.6 = concentrated; < 0.4 = very peaked.
 
 ## Confusion Per SNR
 Saved as `confusion_epoch*.json` with `confusion_per_snr[snr]` an 8×8 matrix (rows=true class, cols=predicted). Diagonal / row sum = per-class accuracy at that SNR. Track low-SNR diagonal improvements against rising low-SNR weights.
 
 ## Healthy Adaptive Pattern
 1. Warmup epochs: uniform weights; accuracy climbs.
 2. First update: modest low-SNR emphasis (≈2–3× uniform).
 3. Subsequent epochs: low-SNR diagonals improve; entropy declines gradually.
 4. Val accuracy stable or rising; high-SNR buckets retain >5% share.
 
 ## Tuning Knobs
 | Param | Effect | Guidance |
 |-------|--------|----------|
 | β | Adapt aggressiveness | 0.2–0.4 typical |
 | ε | Bucket floor | ≈ 0.25 × uniform weight |
 | max_cap | Upper bound per bucket | Start ≈ 0.10–0.15 |
 | lr | Step size | 0.005 stable; reduce if collapse |
 | clipnorm | Stabilize updates | 5–10 early |
 
 ## Advanced Ideas
 - Mixed replay: Reserve % uniform baseline samples.
 - Temperature: Softmax(error/τ) for controllable sharpness.
 - Dynamic β: High early, taper later.
 - Class+SNR curriculum: Switch scope after parity at low SNR.
 
 ## Artifacts
 | Artifact | Path | Notes |
 |----------|------|-------|
 | Weights | `results/adaptive_sampling/<RUN_TAG>/weights_epoch*.json` | Distribution snapshots |
 | Confusion | `results/adaptive_sampling/<RUN_TAG>/confusion_epoch*.json` | Per-SNR confusion matrices |
 | Logs | `results/adaptive_sampling/<RUN_TAG>/logs/` | TensorBoard events |
 | CSV | `results/adaptive_sampling/<RUN_TAG>/squeezenet_sampler_train_log.csv` | Epoch metrics |
 | Model | `models/squeezenet_sampler_<RUN_TAG>.h5` | Best by val accuracy |
 
 ## Quick Inspection Shortcuts
 ```bash
 # Latest weights only (shell trick)
 python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling/<RUN_TAG> --top 12 --entropy | tail -n +1
 
 # Watch weights evolve (simple loop)
 watch -n 30 "python src/adaptive_sampling/inspect_weights.py --dir results/adaptive_sampling/<RUN_TAG> --top 8 --entropy | grep Epoch | tail -n 5"
 ```
 
 ## Checklist Before Declaring Success
 1. Baseline parity validated.
 2. Adaptive (class scope) reaches comparable val accuracy.
 3. Low-SNR per-class diagonals improve vs baseline.
 4. Weight entropy decrease not extreme (ratio >0.4).
 5. No catastrophic forgetting at high SNR.
 6. Metrics & artifacts saved per epoch.
 
 ---
 Update this README as new diagnostics or strategies are added.
Adaptive sampling (sampler-only)

Adds a weighted sampler and epoch-end callback to focus training on hard (class,SNR) buckets.

Run
- Build metadata: python src/adaptive_sampling/dataset_metadata.py
- Train:         python src/adaptive_sampling/train_squeezenet_sampler.py

Baselines: src/baseline_peng, src/baseline_chahil
