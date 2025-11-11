Adaptive sampling (sampler-only)

Adds a weighted sampler and epoch-end callback to focus training on hard (class,SNR) buckets.

Run
- Build metadata: python src/adaptive_sampling/dataset_metadata.py
- Train:         python src/adaptive_sampling/train_squeezenet_sampler.py

Baselines: src/baseline_peng, src/baseline_chahil
