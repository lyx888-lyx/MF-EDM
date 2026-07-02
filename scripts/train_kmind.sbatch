#!/bin/bash
#SBATCH --job-name=train_kmind
#SBATCH --output=%x_%j_%N.out
#SBATCH --partition=comm
#SBATCH --nodelist=coffee
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1

source /opt/anaconda3/bin/activate mf_edm
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1

srun python -u train.py --base-model GRU --dropout 0.4 --lr 0.0001 --batch-size 32 --epochs 60 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset KMIND --norm BN --num_L 2 --num_K 2 --window_spk 1 --window_spk_f 1 --window_dir 1 --window_dir_f 6 --epsilon2 0.1 --epsilon 0.9 --use_speaker i --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
