# model suffixes, display names, x-axis layouts and training seeds.
#
# HARDCODED INTERPOLATION: the models in INTERPOLATE_SUFFIXES were trained with FID-25k and
# path-straightness checks every 50 epochs, all other runs every 80 epochs. To get a mean +- std over
# seeds, their curves are linearly interpolated onto the 80-epoch grid (79, 159, 239, 319, 399).

import torch

TAG = "3seeds"
FORCE_RECOMPUTE = False  # True: ignore all cached results in plots/cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_EPOCH_GAP = 80
SOURCE_EPOCH_GAP = 50

# ---- models ----
BASE = "base_mini64_constant_lr"
IP_CONST = "mini_IP64_constant_constant_lr"
IP_CUSTOM_01 = "mini_IP64_custom_constant_lr"
IP_CUSTOM_02 = "mini_IP64_custom_constant_lr_gamma0is02"
BIP_0025 = "bottleneck_IP_0025_constant_lr"
BIP_005 = "bottleneck_IP_005_constant_lr"
BIP_01 = "bottleneck_IP_01_constant_lr"
BIP_02 = "bottleneck_IP_02_constant_lr"
BIP_04 = "bottleneck_IP_04_constant_lr"
KL_005 = "KL_005"
KL_01 = "KL_01"
KL_025 = "KL_025"
KL_05 = "KL_05"
KL_1 = "KL_1"

# ONLY these suffixes are interpolated from the 50-epoch grid to the 80-epoch grid (HARDCODED, see above)
INTERPOLATE_SUFFIXES = {BASE, IP_CONST, IP_CUSTOM_01}

FROZEN_SPACE = BASE  # KL features of the "frozen" option come from this model's bottleneck

# training seeds: original run + reseed1 + reseed2
SEEDED = {
    BASE: [BASE, f"{BASE}_reseed1", f"{BASE}_reseed2"],
    BIP_02: [BIP_02, f"{BIP_02}_reseed1", f"{BIP_02}_reseed2"],
    KL_05: [KL_05, f"{KL_05}_reseed1", f"{KL_05}_reseed2"],
}

SUFFIX_TO_NAME = {
    BASE: "base",
    IP_CONST: "IP 0.1",
    IP_CUSTOM_01: "IP$^*$ 0.1",
    IP_CUSTOM_02: "IP$^*$ 0.2",
    BIP_0025: "b-IP$^*$ 0.025",
    BIP_005: "b-IP$^*$ 0.05",
    BIP_01: "b-IP$^*$ 0.1",
    BIP_02: "b-IP$^*$ 0.2",
    BIP_04: "b-IP$^*$ 0.4",
    KL_005: "KL 0.05",
    KL_01: "KL 0.1",
    KL_025: "KL 0.25",
    KL_05: "KL 0.5",
    KL_1: "KL 1",
}

# x-axis layout of the b-IP sweep (None = empty slot, keeps the spacing proportional to the b-IP value)
BIP_XTICKS = [
    BIP_0025,  # 0.0025
    BIP_005,   # 0.0050
    None,      # 0.0075
    BIP_01,    # 0.0100
    None, None, None,  # 0.0125 - 0.0175
    BIP_02,    # 0.0200
    None, None, None, None, None, None, None,  # 0.0225 - 0.0375
    BIP_04,    # 0.0400
]

# x-axis layout of the KL-penalty sweep (None = empty slot, spacing proportional to the KL weight)
KL_XTICKS = [
    KL_005,  # 0.05
    KL_01,   # 0.10
    None,    # 0.15
    None,    # 0.20
    KL_025,  # 0.25
    None,    # 0.30
    None,    # 0.35
    None,    # 0.40
    None,    # 0.45
    KL_05,   # 0.50
    None, None, None, None, None, None, None, None, None,  # 0.55-0.95
    KL_1,
]

HEATMAP_SUFFIXES = [BASE, IP_CUSTOM_01, BIP_0025, BIP_005, BIP_01, BIP_02, BIP_04]


def name_of(suffix: str) -> str:
    return SUFFIX_TO_NAME.get(suffix, suffix)