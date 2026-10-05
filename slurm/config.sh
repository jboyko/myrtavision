# Shared settings for the Great Lakes jobs. Sourced by setup.sh and every sbatch file.
# The Slurm account also appears in each #SBATCH --account line (Slurm cannot read variables there).

ACCOUNT=jboyko0
SCRATCH_DIR=/scratch/jboyko_root/${ACCOUNT}/${USER}/myrtavision
ENV_NAME=myrtavision
PRETRAINED="yolo26s-cls.pt yolo26m-cls.pt yolo11m-cls.pt"
LM2_SHA256=549edd12c95b79f2b22c98a38b6c21529a102e6782a123d7c815c5370f08f26b

# Ignore packages in ~/.local, which otherwise leak into the env and break torch.
export PYTHONNOUSERSITE=1
TORCH_INDEX=https://download.pytorch.org/whl/cu126

activate_env() {
    module load python3.11-anaconda/2024.02
    source "$(conda info --base)/etc/profile.d/conda.sh"
    # sbatch copies the submitting shell's environment. If an env was active there,
    # `conda activate` is a no-op and the module's base python (just put first on
    # PATH by module load) wins. Start from a clean conda stack instead.
    while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do
        conda deactivate
    done
    conda activate "$ENV_NAME"
    if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
        echo "python is $(command -v python), not the $ENV_NAME env's ($CONDA_PREFIX)" >&2
        return 1
    fi
}
