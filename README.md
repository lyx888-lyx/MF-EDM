# Fusing Modalities and Emotional Dynamics: A Two-Stage Graph-based Framework for Emotion Recognition in Conversation

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

## 🛠️ Requirements

- Python 3.8.20
- PyTorch 2.4.1
- CUDA 12.1
- torch-geometric 2.6.1
- fvcore, thop, torchinfo

## 🩹 Troubleshooting

- **`ImportError: libnvJitLink.so.12`**: your `torch`/CUDA install is missing `nvidia-nvjitlink-cu12`. Fix with `pip install nvidia-nvjitlink-cu12==12.1.105` (already pinned in `requirements.txt`/`environment.yml`).
- **`ModuleNotFoundError` for packages you're sure are installed, or `KeyError: 'Version'` from `importlib_metadata`**: on shared machines, a user-level `~/.local/lib/pythonX.Y/site-packages` can shadow or conflict with your conda env's packages. Isolate the env by running with `PYTHONNOUSERSITE=1`, e.g.:
  ```bash
  PYTHONNOUSERSITE=1 python -u train.py ...
  ```
  The scripts in `scripts/` already set this.
- **`transformers`/`huggingface_hub` import errors when importing `torch_geometric`**: `torch_geometric.nn` eagerly imports its optional LLM utilities (`torch_geometric.nn.nlp.llm`), which pull in `transformers`. This is installed via `requirements.txt`/`environment.yml`; if you install `torch-geometric` separately (Option B), make sure `transformers` is present too.

## 📁 Datasets
The pre-extracted multimodal features (text, audio, visual) used in this project are adopted from the [M3Net](https://github.com/feiyuchen7/M3NET) project(Chen et al., CVPR 2023), [GraphSmile](https://github.com/lijfrank/GraphSmile) project(Li, Wang, and Zeng, 2025) and . Download multimodal features:
- [IEMOCAP](https://drive.google.com/drive/folders/1s5S1Ku679nlVZQPEfq-6LXgoN1K6Tzmz?usp=drive_link) → Place into the `IEMOCAP_features/` folder  
- [MELD](https://drive.google.com/drive/folders/1GfqY7WNVeCBWoFa_NSTalnaIgyyOVJuC?usp=drive_link) → Place into the `MELD_features/` folder
- K-MIND → Place into the `KMIND_features/` foler
- M3ED → Place into the `M3ED_features/` foler


Download raw datasets(Optional):
- [IEMOCAP](https://sail.usc.edu/iemocap/)
- [MELD](https://github.com/SenticNet/MELD)
- [M3ED](https://github.com/AIM3-RUC/RUCM3ED)


> **K-MIND / M3ED**: place the feature pickle as `KMIND_features/KMIND_features_normalized.pkl` and `M3ED_features/M3ED_features.pkl` respectively. M3ED features follow the [RUCM3ED](https://github.com/AIM3-RUC/RUCM3ED) release; K-MIND is available upon request.

## 🏋️‍♀️ Training

### Train on IEMOCAP
```bash
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size=32 --epochs=60 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=6 --num_K=3 --window_spk=10 --window_spk_f=-1 --window_dir=1 --window_dir_f=-1 --epsilon2=1 --epsilon=1 --use_speaker='bh' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on MELD
```bash
python -u train.py --base-model 'GRU' --dropout=0.4 --lr=0.0001 --batch-size 32 --epochs=6 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=1 --window_spk=3 --window_spk_f=1 --window_dir=8 --window_dir_f=6 --epsilon2=0.1 --epsilon=1.1 --use_speaker='i' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on KMIND
```bash
python -u train.py --base-model GRU --dropout 0.4 --lr 0.0001 --batch-size 32 --epochs 60 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset KMIND --norm BN --num_L 2 --num_K 2 --window_spk 1 --window_spk_f 1 --window_dir 1 --window_dir_f 6 --epsilon2 0.1 --epsilon 0.9 --use_speaker i --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
```

### Train on M3ED
```bash
python -u train.py --base-model GRU --dropout 0.4 --lr 0.001 --batch-size 32 --epochs 60 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset M3ED --norm BN --num_L 2 --num_K 1 --window_spk 3 --window_spk_f 1 --window_dir 4 --window_dir_f 6 --epsilon2 0.1 --epsilon 0.7 --use_speaker i --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
```


## 📉 Experiments
### Ablation Study
Each line shows the arguments changed to disable a specific model component or modality.
```bash
# Table 3 (i) w/o multimodal node (h^m)
--multimodal_node=''
# Table 3 (ii) w/o EIGNN (I^m)
--graph_type='ECG'
# Table 3 (iii) w/o ECGNN (C^m)
--graph_type='EIG'
# Table 4 Modality Configuration
Remove --multi_modal
--modals='l' --directed_edge='l' or --modals='a' --directed_edge='a' or --modals='v' --directed_edge='v'
Do not remove --multi_modal
--modals='la' --directed_edge='la' or --modals='lv' --directed_edge='lv' or --modals='av' --directed_edge='av'
# Appendix Table 5 Effect of Speaker Embedding
--use_speaker=''
# Appendix Table 6 Effect of Intra-Utterance Multi-Edges
--single_edge='intra'
# Appendix Table 7 Effect of Intra-Utterance Directed Edges
--directed_edge='' or --directed_edge='l' or --directed_edge='a' or --directed_edge='v' or --directed_edge='la' or --directed_edge='lv' or --directed_edge='av'
```

### Effect of Hyperparameter
We also evaluate the model’s sensitivity to key hyperparameters:
```bash
# Figure 3: Number of GNN Layers
--num_L=<int>
--num_K=<int>
# Appendix Figure 3: Impact of Residual Scaling εI and εC
# For EIGNN
--epsilon2=<float>
# For ECGNN
--epsilon=<float>
# Appendix Figure 4: Effect of Window Size
# For EIGNN
--window_spk=<int>
--window_spk_f=<int>
# For ECGNN
--window_dir=<int>
--window_dir_f=<int>
```

## 🔧 Argument Highlights

| Argument              | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `--modals`            | Modalities to use for fusion: `'a'` (audio), `'v'` (visual), `'l'` (language). Use combinations like `'avl'`. |
| `--multi_modal`       | Enable multimodal input using text, audio, and visual features.             |
| `--num_L`             | Number of EIGNN layers.                           |
| `--num_K`             | Number of ECGNN layers.                       |
| `--window_spk`        | Past window size for EIGNN.                           |
| `--window_spk_f`      | Future window size for EIGNN.                         |
| `--window_dir`        | Past window size for ECGNN.                           |
| `--window_dir_f`      | Future window size for ECGNN.                         |
| `--epsilon`           | Scaling factor for ECGNN.                           |
| `--epsilon2`          | Scaling factor for EIGNN.                         |
| `--use_speaker`       | Type of speaker embedding in the graph.                         |
| `--multimodal_node`   | Multimodal node in the model: `'EIG'`, `'ECG'`, or `'both'`. |
| `--graph_type`        | Type of graph in the model: `'EIG'`, `'ECG'`, or `'both'`. |
| `--directed_edge`     | Applies directed intra-utterance edges to the multimodal node for selected modalities: `'a'`, `'v'`, `'l'`. |
| `--single_edge`       | Use only a single edge type: `'intra'` (intra-utterance). |
| `--testing`           | Run in test mode using a pre-trained model checkpoint.                      |
| `--overhead`          | Run FLOP/memory/time analysis (only valid when `--testing` is used).        |
---

## 📊 Main Results

Single run with fixed seed 67137 (reproducible via `scripts/`).

| Dataset | weighted-F1 | Accuracy |
|---------|:-----------:|:--------:|
| IEMOCAP | 74.13 | 74.37 |
| MELD    | 67.40 | 68.77 |
| K-MIND  | 74.49 | 77.27 |
| M3ED    | 54.03 | 55.11 |

Reproduce with the scripts in `scripts/` (one per dataset).
