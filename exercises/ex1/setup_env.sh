# Source this file (do not execute it): source setup_env.sh

# Define scratch-backed directories for Singularity cache and temporary build space
export SINGULARITY_TMPDIR=/ibex/user/$USER/tmpdir
export SINGULARITY_CACHEDIR=/ibex/user/$USER/cachedir

# Ensure target directories exist
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
