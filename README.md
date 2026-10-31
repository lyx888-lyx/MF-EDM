# MF-EDM: Graph-based Multimodal Fusion and Emotional Dynamics Modeling for Emotion Recognition in Conversation
<p align="center"><b>🎙️ Interspeech 2026</b></p>

---

<p align="center">
  <b>Sooyeon Hwang</b><sup>1</sup> &nbsp;&middot;&nbsp;
  <b>Eunseong Kwon</b><sup>2</sup> &nbsp;&middot;&nbsp;
  <b>Gahgene Gweon</b><sup>1,2,**</sup>
</p>

<p align="center">
  <sup>1</sup>Interdisciplinary Program in Artificial Intelligence, Seoul National University<br/>
  <sup>2</sup>Department of Intelligence and Information, Seoul National University<br/>
  ✉️ {sooyeoni, eskwon, ggweon}@snu.ac.kr &nbsp;&nbsp;|&nbsp;&nbsp; ** Corresponding author
</p>

---

## Overview
<p align="center">
  <img width="767" height="355" alt="MF-EDM Overview" src="https://github.com/user-attachments/assets/147c4599-2e90-4559-93d4-26a93e4c479b" />
</p>


**Reproduction checklist:** [REPRODUCE.md](REPRODUCE.md) covers exact feature filenames, data sources, environment alternatives, checkpoints, and the original test-selected evaluation caveat. Use `python scripts/check_data.py --dataset IEMOCAP` before training.

## ⚙️ Installation

```bash
# Option A — conda (recommended; exact environment)
conda env create -f environment.yml
conda activate mf_edm

# Option B — pip
conda create -n mf_edm python=3.8.20 -y && conda activate mf_edm
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.6.1
pip install -r requirements.txt
```

**Requirements:** Python 3.8.20 · PyTorch 2.4.1 · CUDA 12.1 · torch-geometric 2.6.1 · fvcore, thop, torchinfo

## 📁 Datasets

The pre-extracted multimodal features (text, audio, visual) used in this project are adopted from the [M3Net](https://github.com/feiyuchen7/M3NET) project (Chen et al., CVPR 2023) and the [GraphSmile](https://github.com/lijfrank/GraphSmile) project (Li, Wang, and Zeng, 2025). Download the pre-extracted multimodal features:
- IEMOCAP → place into the `IEMOCAP_features/` folder
- MELD → place into the `MELD_features/` folder
- K-MIND → place into the `KMIND_features/` folder
- M3ED → place into the `M3ED_features/` folder

Raw dataset sources:
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MELD](https://github.com/SenticNet/MELD)
- [M3ED](https://github.com/AIM3-RUC/RUCM3ED)

> **K-MIND / M3ED**: place the feature pickle as `KMIND_features/KMIND_features_normalized.pkl` and `M3ED_features/M3ED_features.pkl` respectively. M3ED features follow the [RUCM3ED](https://github.com/AIM3-RUC/RUCM3ED) release; K-MIND is available upon request.

## 🏋️ Training
Each dataset has a ready-to-run script under `scripts/` with the paper's hyperparameters (Table 2) already set. Single run with fixed seed 67137.

```bash
bash scripts/train_iemocap.sh
bash scripts/train_meld.sh
bash scripts/train_kmind.sh
bash scripts/train_m3ed.sh
```


## 📊 Main Results

All baselines are reimplemented under the same experimental setting.

| Models              | IEMOCAP w-F1 | IEMOCAP Acc | MELD w-F1 | MELD Acc | K-MIND w-F1 | K-MIND Acc | M3ED w-F1 | M3ED Acc |
|---------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MMGCN               | 65.35 | 65.00 | 59.57 | 60.46 | 72.39 | 77.12 | 46.85 | 50.15 |
| MM-DFN              | 68.27 | 69.07 | 58.69 | 61.57 | 70.21 | 76.19 | 50.90 | 52.46 |
| M3Net               | 69.41 | 69.44 | 65.54 | 66.51 | 73.98 | 76.79 | 50.43 | 52.92 |
| GraphSmile          | 71.36 | 71.60 | 66.61 | 67.59 | 73.62 | 76.37 | 52.37 | 54.51 |
| **MF-EDM (Ours)**   | **74.24** | **74.49** | **67.40** | **68.77** | **74.48** | **77.24** | **55.05** | **56.15** |

## 🔧 Argument Highlights

| Argument            | Description                                                                                                   | Paper notation |
|---------------------|-----------------------------------------------------------------------------------------------------------------|-----------------|
| `--modals`          | Modalities to use for fusion: `'a'` (audio), `'v'` (visual), `'l'` (language). Use combinations like `'avl'`.   | {t, a, v} |
| `--multi_modal`     | Enable multimodal input using text, audio, and visual features.                                                 | — |
| `--num_L`           | Number of EIGNN layers.                                                                                          | L<sup>I</sup> |
| `--num_K`           | Number of ECGNN layers.                                                                                         | L<sup>C</sup> |
| `--window_spk`      | Past window size for EIGNN.                                                                                      | P<sup>I</sup> |
| `--window_spk_f`    | Future window size for EIGNN.                                                                                   | F<sup>I</sup> |
| `--window_dir`      | Past window size for ECGNN.                                                                                      | P<sup>C</sup> |
| `--window_dir_f`    | Future window size for ECGNN.                                                                                   | F<sup>C</sup> |
| `--epsilon`         | Residual scaling factor for ECGNN.                                                                               | ε<sub>C</sub> |
| `--epsilon2`        | Residual scaling factor for EIGNN.                                                                               | ε<sub>I</sub> |
| `--use_speaker`     | Type of speaker embedding in the graph.                                                                          | speaker embedding |
| `--multimodal_node` | Multimodal anchor node used in the model: `'EIG'`, `'ECG'`, or `'both'`.                                        | h<sup>m</sup> |
| `--graph_type`      | Which Stage-2 GNN(s) feed the classifier: `'EIG'`, `'ECG'`, or `'both'`.                                        | EIGNN / ECGNN |
| `--directed_edge`   | Applies directed intra-utterance edges into the multimodal anchor node for selected modalities: `'a'`, `'v'`, `'l'`. | {h<sup>x</sup> → h<sup>m</sup>} |
| `--single_edge`     | Restrict to a single edge type: `'intra'` (intra-utterance only).                                               | Ek edge types |
| `--testing`         | Run in test mode using a pre-trained model checkpoint.                                                          | — |
| `--overhead`        | Run FLOP / memory / time analysis (only valid with `--testing`).                                                | — |

## 📚 Citation

```bibtex
@inproceedings{hwang2026mfedm,
  title     = {{MF-EDM}: Graph-based Multimodal Fusion and Emotional Dynamics Modeling for Emotion Recognition in Conversation},
  author    = {Hwang, Sooyeon and Kwon, Eunseong and Gweon, Gahgene},
  booktitle = {Interspeech},
  year      = {2026}
}
```
