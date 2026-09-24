# Robustness of Speech Codecs Beyond Speech

Reproducibility repository for the paper *"Robustness of Speech Codecs Beyond Speech: Quantifying and Recovering from Coding-Induced Degradation in Audio Event Classification"* (submitted to *Speech Communication*).

## Summary

Acoustic classifiers are increasingly applied to audio that has passed through a
communication channel. This work quantifies how real speech coding degrades
acoustic event classification, separates the loss into bandwidth and
codec-specific components, shows the effect across three codecs and two corpora,
and demonstrates a simple training remedy.

Headline results (clean-trained ResNet-50, macro-F1):

- AMR-NB at 4.75 kbit/s drops UrbanSound8K macro-F1 from 0.786 to 0.370
  (a 41.6-point loss); the drop scales with bitrate (Spearman rho = 0.905).
- A band-limiting control attributes about 48% of the loss to bandwidth
  restriction and 52% to codec-specific processing (p_Holm = 0.006).
- The effect generalises to Opus and AMR-WB, and to a second corpus (ESC-50,
  41.7-point drop). Per-class vulnerability tracks high-frequency reliance
  across 50 ESC-50 classes (rho = 0.40, p = 0.004).
- Codec-aware augmentation recovers 73-89% of the loss across the three codecs
  and reaches statistical equivalence with the clean baseline at 12.2 kbit/s.
  It far exceeds generic augmentation (12% recovery).

## Repository layout

```
notebooks/      The notebooks that generate every result
results/        The JSON result files produced by the notebooks
figures/        The six manuscript figures plus the graphical abstract
```

## How to reproduce

Each notebook is self-contained and was run on Kaggle with a single T4 GPU. To
reproduce a result:

1. Open the relevant notebook on Kaggle (or any Jupyter environment with a GPU).
2. Add the three public datasets as inputs (see *Datasets* below).
3. Enable Internet (needed once, to install the codec tools) and a GPU.
4. Run all cells. Every fold is checkpointed, so an interrupted run resumes
   without recomputation.

To regenerate all six figures from the stored JSON results without re-training:

```
python notebooks/regenerate_all_figures.py
```

(with the JSON files in `results/`; figures are written to `results/figs/`).

## Notebook-to-result map

| Notebook | Produces | Manuscript element |
|---|---|---|
| `1_main_experiment.ipynb`        | `regimes.json`, `bitrate_sweep.json` | Tables 1-2, Figs 1-2 |
| `2_bandlimit_control.ipynb`      | `bandlimit_comparison.json`, `bandlimit_statistics.json`, `acoustic_verification.json` | Table 3, Figs 3, 5 |
| `3_second_codec.ipynb`           | `second_codec_results.json` | Table 4, Fig 4 |
| `4_supporting_experiments.ipynb` | `experiment2_generic_aug.json`, `experiment3_esc50.json`, `experiment3_frequency_dependence.json` | Tables 5-6, Fig 6 |
| `5_acoustic_characterisation.ipynb` | `mismatch_env_vs_speech.json`, `acoustic_profile.json` | Section 4.6 acoustic mismatch |
| `regenerate_all_figures.py`      | all six figure PDFs/PNGs | Figs 1-6 |

## Datasets

The corpora are public and are not redistributed here.

- UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html
- ESC-50: https://github.com/karolpiczak/ESC-50
- Speech Commands (v0.02): https://www.tensorflow.org/datasets/catalog/speech_commands

## Environment

See `requirements.txt`. The experiments used Python 3.11, PyTorch with a CUDA
T4 GPU, and `ffmpeg` with the `libopencore-amrnb`, `libvo-amrwbenc`, and
`libopus` encoders. A fixed random seed (42) is used throughout.

## Citation

If you use this code or the results, please cite the work (https://doi.org/10.5281/zenodo.22899468)

## License

Released under the MIT License. See `LICENSE`.
