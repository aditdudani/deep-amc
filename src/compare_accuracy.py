import json
import os
import matplotlib.pyplot as plt

BASELINE_JSON = 'results_local/squeezenet/accuracy_by_snr_squeezenet.json'
GOOGLENET_JSON = 'results_local/googlenet/accuracy_by_snr.json'
ADAPTIVE_V1_JSON = 'results_local/adaptive_sampling/adaptive_v1/accuracy_by_snr_squeezenet.json'
ADAPTIVE_V2_JSON = 'results_local/adaptive_sampling/adaptive_v2/accuracy_by_snr_squeezenet.json'
OUT_PATH = 'results_local/compare/accuracy_vs_snr2.png'

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

def load(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    acc_map = data.get('accuracy_by_snr') or data.get('accuracy_by_snr_squeezenet')
    if acc_map is None:
        raise ValueError(f'No accuracy map in {json_path}')
    snrs = sorted(int(k) for k in acc_map.keys())
    accs = [float(acc_map[str(s)] if str(s) in acc_map else acc_map[s]) for s in snrs]
    return snrs, accs

snr_b, acc_b = load(BASELINE_JSON)
snr_g, acc_g = load(GOOGLENET_JSON)
snr_a1, acc_a1 = load(ADAPTIVE_V1_JSON)
snr_a2, acc_a2 = load(ADAPTIVE_V2_JSON)

plt.figure(figsize=(7.5, 4.5))
plt.plot(snr_b, acc_b, marker='o', label='Baseline SqueezeNet')
plt.plot(snr_g, acc_g, marker='o', label='Baseline GoogLeNet')
plt.plot(snr_a1, acc_a1, marker='o', label='Adaptive SqueezeNet v1')
plt.plot(snr_a2, acc_a2, marker='o', label='Adaptive SqueezeNet v2')
plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs SNR')
plt.grid(alpha=0.3)
plt.ylim(0.35, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH)
print(f'Saved {OUT_PATH}')
