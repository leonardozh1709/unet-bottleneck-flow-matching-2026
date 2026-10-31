![CIFAR-10 samples from across the project.](_docs/misc_generated_samples.png)

# Experiments on the U-Net Bottleneck in Flow Matching

## An Empirical Deep Learning Study

All training, sampling and evaluation code used in the paper can be found here.
A PDF of the paper itself can be found in `./_docs`.

All model weights are in the outputs folder in this google drive link:
Unzip that folder into the project root to load and use those models.

## Setup/Usage

`pip install -r requirements.txt`

For inference, use SAMPLING_MASTER.py by running the project as a module:
`python -m main.SAMPLING_MASTER`
You can chose a model suffix at the top of the file.

## Repository structure

```
├── main/
│   ├── SAMPLING_MASTER.py
│   ├── training.py
│   └── TRAINING_MASTER.py
├── metrics/
│   ├── bottleneck_swap_test.py
│   ├── erf.py
│   ├── exposure_bias_kl.py
│   ├── FID.py
│   ├── inter_channel_correlation.py
│   ├── modelderivative.py
│   ├── pathstraightness.py
│   ├── plot_all_pathstraightness.py
│   ├── plot_multiple_FID50k.py
│   └── train_vs_test_loss.py
├── ModelAnalyzerV2/
│   └── ... (modelanalyzer code)
├── one_shots/
│   ├── autotuningtrainingparams.py
│   ├── calibrate_fresh_base_mini_loss.py
│   ├── calibrate_kl_penalty_scale.py
│   ├── checkconfig.py
│   ├── findbestcfg.py
│   ├── sweep_kl_gradient_ratio.py
│   ├── test_interpolation.py
│   ├── update_saved_configs.py
│   ├── visualizecfg.py
│   └── __init__.py
├── outputs/
│   └── ... (model weights)
├── Utils/
│   ├── config.py
│   ├── data.py
│   ├── loadmodel.py
│   ├── paths.py
│   ├── plotimages.py
│   ├── registry.py
│   ├── sample.py
│   ├── seeds.py
│   ├── UNet.py
│   ├── UNet_max.py
│   └── __init__.py
├── _docs/
│   ├── euler_base_mini64_constant_lr.png
│   ├── euler_bottleneck_IP_02_constant_lr_reseed2.png
│   ├── euler_KL_05_reseed1.png
│   └── misc generated samples.png
├── .gitignore
├── README.md
├── requirements.txt
└── structure.txt
```