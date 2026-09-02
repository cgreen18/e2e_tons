#!/bin/bash -l
# Build both ASTRA analytical binaries on a compute node.
#
# `module` is a shell function that only exists in a login shell, hence the -l
# in the shebang. The module load is NOT allowed to fail silently: without
# protobuf on the library path the binaries die with exit 127 at run time.
#SBATCH -A mithuna
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 02:00:00
set -euo pipefail
cd /home/green456/e2e_tons
module load protobuf/3.18.0
# The system GCC 8 cannot link std::filesystem; use the spack GCC 12 the
# analytical backend expects.
export CC=/apps/spack/negishi/apps/gcc/12.2.0-gcc-8.5.0-27t2zcj/bin/gcc
export CXX=/apps/spack/negishi/apps/gcc/12.2.0-gcc-8.5.0-27t2zcj/bin/g++
echo "host=$(hostname) $(date -Is)"
simul/astra-sim/build/astra_analytical/build.sh
for b in simul/astra-sim/build/astra_analytical/build/bin/*; do
  if ldd "$b" | grep -q "not found"; then
    echo "FATAL: unresolved shared libraries in $b" >&2; ldd "$b" | grep "not found" >&2; exit 1
  fi
done
ls -la simul/astra-sim/build/astra_analytical/build/bin/
echo "build OK $(date -Is)"
