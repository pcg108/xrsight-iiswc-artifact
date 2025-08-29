#!/bin/bash

# -------------------------
# Configuration
# -------------------------
CY_DIR="$(pwd)/xrsight-chipyard"
ROOT_DIR="$(pwd)"

# -------------------------
# Setup
# -------------------------
cd "$CY_DIR/sims/firesim" || { echo "Failed to cd to FireSim dir"; exit 1; }

# firesim setup
cd ~/.ssh
ssh-agent -s > AGENT_VARS
source AGENT_VARS
ssh-add firesim.pem
cd -
source sourceme-manager.sh || { echo "Failed to source FireSim manager"; exit 1; }

# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# copy ubuntu-base-bin over to firemarshal directory
cp $ROOT_DIR/firesim-settings/ubuntu-base-bin $CY_DIR/software/firemarshal/images/firechip/ubuntu-base/ubuntu-base-bin || { echo "Failed to copy ubuntu-base-bin"; exit 1; }

# move the ubuntu-base-vio to the firechip directory
if [ -f "$ROOT_DIR/temp/ubuntu-base-vio.img" ]; then
    echo "Moving ubuntu-base-vio.img to firechip directory..."
    mv $ROOT_DIR/temp/ubuntu-base-vio.img $CY_DIR/software/firemarshal/images/firechip/ubuntu-base/ || { echo "Failed to move ubuntu-base-vio.img"; exit 1; }
else
    echo "ubuntu-base-vio.img not found in temp directory."
fi

cp $ROOT_DIR/firesim-settings/config_build_recipes.yaml $CY_DIR/sims/firesim-staging/config_build_recipes.yaml || { echo "Failed to copy config_build_recipes.yaml"; exit 1; }
cp $ROOT_DIR/firesim-settings/config_hwdb.yaml $CY_DIR/sims/firesim-staging/config_hwdb.yaml || { echo "Failed to copy config_hwdb.yaml"; exit 1; }

set -euo pipefail
YAML_FILE=${CY_DIR}/sims/firesim-staging/config_build_recipes.yaml
NEW_PATH="${CY_DIR}"
# Replace <placeholder> with $NEW_PATH
sed -i "s|<placeholder>|$NEW_PATH|g" "$YAML_FILE" || { echo "Failed to update config_build_recipes.yaml"; exit 1; }

YAML_FILE="${CY_DIR}/sims/firesim-staging/config_hwdb.yaml"
NEW_PATH="${ROOT_DIR}"
# Replace <placeholder> with $NEW_PATH
sed -i "s|<placeholder>|$NEW_PATH|g" "$YAML_FILE" || { echo "Failed to update config_hwdb.yaml"; exit 1; }

# copy the ubuntu-base.json to the firesim directory
cp $ROOT_DIR/firesim-settings/vio-saturn/ubuntu-base.json deploy/workloads/ubuntu-base.json || { echo "Failed to copy ubuntu-base.json"; exit 1; }

# copy the config_runtime.yaml to the firesim directory
cp $ROOT_DIR/firesim-settings/vio-saturn/config_runtime.yaml $CY_DIR/sims/firesim/deploy/config_runtime.yaml || { echo "Failed to copy config_runtime.yaml"; exit 1; }

firesim infrasetup -a "${CY_DIR}/sims/firesim-staging/config_hwdb.yaml" \
                    -r "${CY_DIR}/sims/firesim-staging/config_build_recipes.yaml" 

# -------------------------
# Step 1: run FireSim workload
# -------------------------
echo "Launching FireSim workload..."
firesim runworkload -a "${CY_DIR}/sims/firesim-staging/config_hwdb.yaml" \
                    -r "${CY_DIR}/sims/firesim-staging/config_build_recipes.yaml" > /dev/null 2>&1 &

echo "Setup complete. FireSim workload launched."
echo "Run the monitor script to track progress and handle cleanup."
exit 0