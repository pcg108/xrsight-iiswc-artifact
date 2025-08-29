#!/bin/bash

# -------------------------
# Configuration
# -------------------------
SESSION="fsim0"
SCROLLBACK=1000000
INTERVAL=15   # seconds between checks
OUTDIR="$(pwd)/screen_dumps"
CY_DIR="$(pwd)/xrsight-chipyard"
ROOT_DIR="$(pwd)"

mkdir -p "$OUTDIR"
chmod u+w "$OUTDIR"

# -------------------------
# Commands to run after login
# -------------------------
COMMANDS=(
    "./rvvTest-zvl256 3 8 8 8 1 1 0 0 0 20"
    "./rvvTest-zvl256 3 16 16 16 1 1 0 0 0 20"
    "./rvvTest-zvl256 3 32 32 32 1 1 0 0 0 20"
    "./rvvTest-zvl256 3 64 64 64 1 1 0 0 0 20"
    "./rvvTest-zvl256 3 128 128 128 1 1 0 0 0 20"
    "./rvvTest-zvl256 3 256 256 256 1 1 0 0 0 20"
    "./rvvTest-zvl256 1 32 1 1 0 20"
    "./rvvTest-zvl256 1 64 1 1 0 20"
    "./rvvTest-zvl256 1 128 1 1 0 20"
    "./rvvTest-zvl256 1 256 1 1 0 20"
    "./rvvTest-zvl256 1 512 1 1 0 20"
    "cd OpenVINS"
    "./run_illixr_msckf-vcv-256 data/mav0/cam0/data.csv data/mav0/cam1/data.csv data/mav0/imu0/data.csv data/mav0/cam0/data data/mav0/cam1/data"
    "cat ov_msckf_timing.txt"
)

# -------------------------
# Cleanup function
# -------------------------
cleanup() {
    trap - EXIT INT

    echo "[$(date +%H:%M:%S)] Cleaning up..."

    # Kill FireSim safely
    echo "Killing FireSim workloads..."
    cd "$CY_DIR/sims/firesim" || { echo "Failed to cd to FireSim dir"; exit 1; }
    cd ~/.ssh && ssh-agent -s > AGENT_VARS && source AGENT_VARS && ssh-add firesim.pem && cd -
    source sourceme-manager.sh || { echo "Failed to source FireSim manager"; exit 1; }
    firesim kill -a "${CY_DIR}/sims/firesim-staging/config_hwdb.yaml" \
                  -r "${CY_DIR}/sims/firesim-staging/config_build_recipes.yaml"
    
    exit 0
}
trap cleanup EXIT INT

# -------------------------
# Function to run commands sequentially
# -------------------------
run_commands() {
    echo "[$(date +%H:%M:%S)] Starting command execution..."
    
    for i in "${!COMMANDS[@]}"; do
        cmd="${COMMANDS[$i]}"
        cmd_num=$((i + 1))
        total_cmds=${#COMMANDS[@]}
        
        echo "[$(date +%H:%M:%S)] Running command $cmd_num/$total_cmds: $cmd"
        
        # Send the command to the screen session
        screen -S "$SESSION" -X stuff "$cmd"'\n'
        
        # Wait for command to complete by monitoring the prompt
        echo "[$(date +%H:%M:%S)] Waiting for command to complete..."
        
        while true; do
            sleep 5
            TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
            TMPFILE="$OUTDIR/cmd_check_$TIMESTAMP.txt"
            
            # Get screen buffer content
            screen -S "$SESSION" -X hardcopy -h "$TMPFILE"
            sleep 2  # Allow time for hardcopy to complete
            
            # Check if command completed (looking for prompt)
            if [ -f "$TMPFILE" ]; then
                # Check for either root@ubuntu:~# or being in OpenVINS directory
                if tail -5 "$TMPFILE" | grep -q "root@ubuntu:~#" || \
                   tail -5 vio_saturn.txt | grep -q "root@ubuntu:.*OpenVINS.*#"; then
                    echo "[$(date +%H:%M:%S)] Command $cmd_num completed"
                    rm -f "$TMPFILE"
                    break
                fi
            fi
            
            # Clean up temp file
            rm -f "$TMPFILE"
            
            # Add a longer sleep to avoid overwhelming the system
            sleep 10
        done
        
        # Add a small delay between commands
        sleep 2
    done
    
    echo "[$(date +%H:%M:%S)] All commands completed successfully!"
}

# -------------------------
# Step 1: wait for fsim0 screen session
# -------------------------
echo "Waiting for screen session '$SESSION'..."
while true; do
    if screen -list | grep -q "\.${SESSION}[[:space:]]"; then
        echo "Screen session '$SESSION' found"
        break
    fi
    sleep 2
done
echo "Screen session '$SESSION' is up"

# -------------------------
# Step 2: set scrollback
# -------------------------
screen -S "$SESSION" -X scrollback $SCROLLBACK

# -------------------------
# Step 3-5: monitor login and run commands
# -------------------------
found_login=false
found_password=false
logged_in=false
commands_executed=false

while true; do
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    TMPFILE="$OUTDIR/dump_$TIMESTAMP.txt"

    # Get screen buffer content
    screen -S "$SESSION" -X hardcopy -h "$TMPFILE"
    sleep 5  # Allow time for hardcopy to complete
    
    # Check if hardcopy was successful
    if [ ! -f "$TMPFILE" ]; then
        echo "[$TIMESTAMP] Warning: Failed to create screen dump"
        sleep $INTERVAL
        continue
    fi

    # Handle login sequence with debugging
    if [ "$found_login" = false ] && grep -q "ubuntu login:" "$TMPFILE"; then
        echo "[$TIMESTAMP] Saw login prompt → sending 'root'"
        screen -S "$SESSION" -X stuff 'root\n'
        found_login=true
    fi

    if [ "$found_login" = true ] && [ "$found_password" = false ] && grep -q "Password:" "$TMPFILE"; then
        echo "[$TIMESTAMP] Saw password prompt → sending 'firesim'"
        screen -S "$SESSION" -X stuff 'firesim\n'
        found_password=true
    fi

    if [ "$found_password" = true ] && [ "$logged_in" = false ] && grep -q "root@ubuntu:~#" "$TMPFILE"; then
        echo "[$TIMESTAMP] Logged in as root"
        logged_in=true
    fi

    # Debug: Show current login state
    echo "[$TIMESTAMP] Login state: found_login=$found_login, found_password=$found_password, logged_in=$logged_in"

    # Run the commands once logged in
    if [ "$logged_in" = true ] && [ "$commands_executed" = false ]; then
        echo "[$TIMESTAMP] Starting command execution sequence..."
        run_commands
        commands_executed=true
        echo "[$TIMESTAMP] All commands completed successfully. Initiating cleanup..."
        screen -S "$SESSION" -X hardcopy -h "$OUTDIR/vio_saturn.txt"
        cleanup
    elif [ "$logged_in" = false ]; then
        echo "[$TIMESTAMP] Not logged in yet"
    fi

    # Clean up old temp files to save space (keep only last 5)
    ls -t "$OUTDIR"/dump_*.txt 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null

    sleep $INTERVAL
done