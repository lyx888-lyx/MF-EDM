#!/bin/bash
#SBATCH --job-name=train_meld
#SBATCH --output=%x_%j_%N.out
#SBATCH --partition=comm
#SBATCH --nodelist=coffee
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1

source /opt/anaconda3/bin/activate m3net_default
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1

srun python -u train.py --base-model GRU --dropout 0.4 --lr 0.0001 --batch-size 32 --epochs 6 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset MELD --norm BN --num_L 1 --num_K 1 --window_spk 3 --window_spk_f 1 --window_dir 8 --window_dir_f 6 --epsilon2 0.1 --epsilon 1.1 --use_speaker i --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
