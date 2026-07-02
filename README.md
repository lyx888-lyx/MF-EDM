# MF-EDM: Graph-based Multimodal Fusion and Emotional Dynamics Modeling for Emotion Recognition in Conversation

<p align="center">
  <b>Sooyeon Hwang</b><sup>1</sup> &nbsp;&middot;&nbsp;
  <b>Eunseong Kwon</b><sup>2</sup> &nbsp;&middot;&nbsp;
  <b>Gahgene Gweon</b><sup>1,2,**</sup>
</p>

<p align="center">
  <sup>1</sup>Interdisciplinary Program in Artificial Intelligence, Seoul National University<br/>
  <sup>2</sup>Department of Intelligence and Information, Seoul National University<br/>
  <b>Cognitive Computing Lab, Seoul National University</b><br/>
  ✉️ {sooyeoni, eskwon, ggweon}@snu.ac.kr &nbsp;&nbsp;|&nbsp;&nbsp; ** Corresponding author
</p>

---

## Overview
<p align="center">
  <img width="767" height="355" alt="MF-EDM Overview" src="https://github.com/user-attachments/assets/147c4599-2e90-4559-93d4-26a93e4c479b" />
</p>


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

### Train on IEMOCAP
```bash
python -u train.py --base-model 'GRU' --dropout 0.5 --lr 0.0001 --batch-size=32 --epochs=60 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='IEMOCAP' --norm BN --num_L=6 --num_K=3 --window_spk=10 --window_spk_f=-1 --window_dir=1 --window_dir_f=-1 --epsilon2=1 --epsilon=1 --use_speaker='bh' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on MELD
```bash
python -u train.py --base-model 'GRU' --dropout=0.4 --lr=0.0001 --batch-size 32 --epochs=6 --multi_modal --mm_fusion_mthd='concat_DHT' --modals='avl' --Dataset='MELD' --norm BN --num_L=1 --num_K=1 --window_spk=3 --window_spk_f=1 --window_dir=8 --window_dir_f=6 --epsilon2=0.1 --epsilon=1.1 --use_speaker='i' --multimodal_node='both' --graph_type='both' --directed_edge='avl' --single_edge=''
```

### Train on K-MIND
```bash
python -u train.py --base-model GRU --dropout 0.4 --lr 0.0001 --batch-size 32 --epochs 60 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset KMIND --norm BN --num_L 2 --num_K 2 --window_spk 1 --window_spk_f 1 --window_dir 1 --window_dir_f 6 --epsilon2 0.1 --epsilon 0.9 --use_speaker i --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
```

### Train on M3ED
```bash
python -u train.py --base-model GRU --dropout 0.4 --lr 0.001 --batch-size 32 --epochs 60 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset M3ED --norm BN --num_L 2 --num_K 1 --window_spk 3 --window_spk_f 1 --window_dir 4 --window_dir_f 6 --epsilon2 0.1 --epsilon 0.7 --use_speaker i --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
```

Reproduce with the scripts in `scripts/` (one per dataset). Single run with fixed seed 67137.

## 📊 Main Results

All baselines are reimplemented under the same experimental setting.

h>w-F1</th><th>Acc</th>
<th>w-F1</th><th>Acc</th>
<th>w-F1</th><th>Acc</th>
</tr>
</thead>
<tbody>
<tr><td>MMGCN</td><td>65.35</td><td>65.00</td><td>59.57</td><td>60.46</td><td>72.39</td><td>77.12</td><td>46.85</td><td>50.15</td></tr>
<tr><td>MM-DFN</td><td>68.27</td><td>69.07</td><td>58.69</td><td>61.57</td><td>70.21</td><td>76.19</td><td>50.90</td><td>52.46</td></tr>
<tr><td>M3Net</td><td>69.41</td><td>69.44</td><td>65.54</td><td>66.51</td><td>73.98</td><td>76.79</td><td>50.43</td><td>52.92</td></tr>
<tr><td>GraphSmile</td><td>71.36</td><td>71.60</td><td>66.61</td><td>67.59</td><td>73.62</td><td>76.37</td><td>52.37</td><td>54.51</td></tr>
<tr><td><b>MF-EDM (Ours)</b></td><td><b>74.24</b></td><td><b>74.49</b></td><td><b>67.40</b></td><td><b>68.77</b></td><td><b>74.48</b></td><td><b>77.24</b></td><td><b>55.05</b></td><td><b>56.15</b></td></tr>
</tbody>
</table>

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
