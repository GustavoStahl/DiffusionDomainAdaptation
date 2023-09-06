#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB 
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --job-name syn2real
#SBATCH --output slurm.out
#SBATCH --error slurm.err
#SBATCH --mail-user=gustavo.stahl@kaust.edu.sa
#SBATCH --mail-type=ALL
# #SBATCH --reservation=A100
#SBATCH --gres=gpu:v100:1

#SBATCH --signal=B:USR1@600

# Load environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module add cuda/11.0.1
source /home/stahlg/.bashrc
conda activate syn2real

sig_handler()
{
echo "Force closing connection" && exit 0
}

# get tunneling info 
export XDG_RUNTIME_DIR="./tmp" node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=1593
echo $node pinned to port $port 
# print tunneling instructions 

echo -e " 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa 

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop " 

trap 'sig_handler' USR1 

# Run Jupyter 
jupyter lab .. --no-browser --port=${port} --port-retries=50 --ip=${node} &
wait
