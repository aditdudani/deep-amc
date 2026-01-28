import json

# Content from data/classes-fixed.json
hdf5_classes = [
  "OOK", "4ASK", "8ASK",
  "BPSK", "QPSK", "8PSK",
  "16PSK", "32PSK", "16APSK",
  "32APSK", "64APSK", "128APSK",
  "16QAM", "32QAM", "64QAM",
  "128QAM", "256QAM", "AM-SSB-WC",
  "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
  "FM", "GMSK", "OQPSK"
]

print("--- HDF5 Order (Index -> Class) ---")
for i, cls in enumerate(hdf5_classes):
    print(f"{i}: {cls}")

model_classes = sorted(hdf5_classes)
print("\n--- Model Output Order (Alphabetical) (Class -> Index) ---")
for i, cls in enumerate(model_classes):
    print(f"{cls}: {i}")

print("\n--- Mapping Dictionary (HDF5 Index -> Model Index) ---")
mapping = {}
print("{")
for hdf5_idx, cls in enumerate(hdf5_classes):
    model_idx = model_classes.index(cls)
    mapping[hdf5_idx] = model_idx
    print(f"    {hdf5_idx}: {model_idx},  # {cls} -> {model_idx} ({model_classes[model_idx]})")
print("}")
