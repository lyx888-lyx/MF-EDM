# MF-EDM reproduction checklist (engineering notes)

This is a **baseline-oriented** guide. It preserves the authors' model and
training command arguments, and documents evaluation-protocol limitations.

## 1. Environment

The upstream README specifies Python 3.8.20, PyTorch 2.4.1, CUDA 12.1, and
PyTorch Geometric 2.6.1. The upstream `environment.yml` also pins
`numpy==1.19.5` in pip dependencies and does **not** fully enumerate all of
the packages from `requirements.txt`. For a practical modern environment,
the commands below are a **proposed alternative, not an experimentally
verified exact replica**:

```bash
conda create -n mf_edm python=3.10 -y
conda activate mf_edm
python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/
# Use the appropriate wheel for your NVIDIA driver / CUDA setup.
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-geometric==2.6.1 numpy==1.26.4 pandas==2.2.3 \
    scipy==1.14.1 scikit-learn==1.5.2 fvcore thop torchinfo transformers \
    -i https://mirrors.aliyun.com/pypi/simple/
python -c "import torch, torch_geometric; print(torch.__version__, torch.cuda.is_available(), torch_geometric.__version__)"
```

Confirm `nvidia-smi` and `torch.cuda.is_available()` on your machine before
running full training. GPU availability has not been tested by these notes.

## 2. Get feature files, not just raw videos/audio

Download compatible, *pre-extracted* features, then place them at exactly:

| Dataset | Expected files |
| --- | --- |
| IEMOCAP | `IEMOCAP_features/IEMOCAP_features.pkl`; `IEMOCAP_features/iemocap_features_roberta.pkl` |
| MELD | `MELD_features/MELD_features_raw1.pkl`; `MELD_features/meld_features_roberta.pkl` |
| K-MIND | `KMIND_features/KMIND_features_normalized.pkl` (author says available upon request) |
| M3ED | `M3ED_features/M3ED_features.pkl` |

Upstream resources:

- M3NET feature collection for IEMOCAP/MELD:
  https://www.dropbox.com/sh/4b21lympehwdg4l/AADXMURD5uCECN_pvvJpCAy9a?dl=0
- GraphSmile features (check actual filenames/schema before using them):
  https://drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi?usp=drive_link
- RUCM3ED's released M3ED feature source:
  https://github.com/AIM3-RUC/RUCM3ED (README lists a Baidu Netdisk link)
- IEMOCAP raw data: https://sail.usc.edu/iemocap/
- MELD dataset: https://github.com/SenticNet/MELD

**Do not automatically rename an incompatible pickle.** The feature dimensions,
tuple structure, split keys, speaker IDs, and label order must match
`dataloader.py`. Python pickle can execute code; load only trusted files.

Run a dependency-free existence check first:

```bash
python scripts/check_data.py --dataset all
# or start with one easier-to-obtain benchmark:
python scripts/check_data.py --dataset IEMOCAP
python -m unittest discover -s tests -v
```

## 3. Run the baseline

Run from the repo root. The scripts already set the released dataset-specific
hyperparameters and a fixed seed in `train.py`:

```bash
bash scripts/train_iemocap.sh
bash scripts/train_meld.sh
bash scripts/train_kmind.sh
bash scripts/train_m3ed.sh
```

Do not compare metrics across a different preprocessing pipeline or label map.
The authors' README reports these weighted-F1/accuracy (%) pairs:

| Dataset | Weighted F1 | Accuracy |
| --- | ---: | ---: |
| IEMOCAP | 74.24 | 74.49 |
| MELD | 67.40 | 68.77 |
| K-MIND | 74.48 | 77.24 |
| M3ED | 55.05 | 56.15 |

The checkpoint is saved at `best_model_<DATASET>.pth` each time the
**test weighted F1 improves**. This fork fixes missing M3ED checkpoint saving
and the upstream hardcoded thresholds, but does not change its model
architecture or the reported score selection logic.

## 4. Evaluation caveat (important)

In the original `train.py`, the **test set is evaluated every epoch** and
`best_fscore` is selected based on **test** weighted F1. The IEMOCAP/MELD
training scripts also pass `valid=0.0`, so they have no validation samples.
This is *not* a strictly held-out evaluation protocol, and choosing epochs by
test F1 can inflate reported results. For baseline comparison with the released
code, first reproduce the original setting and explicitly label it
`author-code / test-selected`.

For any follow-up research/ablation, build a validation-selected protocol:
choose epoch by validation F1 (or validation loss), lock the epoch/checkpoint,
and evaluate once on test; keep the two types of results in separate tables.
Do not claim validation-selected scores are the same experiment as the
authors' released baseline.

## 5. Known limitations and next checks

- No feature pickle or pretrained MF-EDM checkpoint is shipped in this git repo.
- A FOUND result from `check_data.py` checks only existence and file size,
  **not** the schema/dimensions or successful training.
- The current experimental runner has no CLI data-root override: files must
  be placed at the paths above (or you must deliberately adapt the loader).
- Verify each feature tuple and smoke-test one batch before a full GPU run.
- Repeat runs and record driver, GPU, package versions, seed, checkpoint epoch,
  best test F1, and accuracy to make comparisons auditable.
