#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1

python -u train.py --base-model GRU --dropout 0.4 --lr 0.0001 --batch-size 32 --epochs 60 \
  --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset KMIND --norm BN \
  --num_L 2 --num_K 2 --window_spk 1 --window_spk_f 1 --window_dir 1 --window_dir_f 6 \
  --epsilon2 0.1 --epsilon 0.9 --use_speaker i --multimodal_node both --graph_type both \
  --directed_edge avl --single_edge ''
