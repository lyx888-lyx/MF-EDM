#!/bin/bash
#SBATCH --job-name=train_iemocap
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

srun python -u train.py --base-model GRU --dropout 0.5 --lr 0.0001 --batch-size 32 --epochs 60 --multi_modal --mm_fusion_mthd concat_DHT --modals avl --Dataset IEMOCAP --norm BN --num_L 6 --num_K 3 --window_spk 10 --window_spk_f -1 --window_dir 1 --window_dir_f -1 --epsilon2 1 --epsilon 1 --use_speaker bh --multimodal_node both --graph_type both --directed_edge avl --single_edge ''
