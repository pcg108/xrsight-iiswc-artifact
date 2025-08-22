#!/bin/bash

# -------------------------
# Configuration
# -------------------------
SESSION="fsim0"
SCROLLBACK=100000
INTERVAL=15   # seconds between checks
OPENVINS_PREFIX="openvins:"
OPENVINS_COUNT=190
OUTDIR="$(pwd)/screen_dumps"
CY_DIR="$(pwd)/xrsight-chipyard"
ROOT_DIR="$(pwd)"

mkdir -p "$OUTDIR"
chmod u+w "$OUTDIR"

# -------------------------
# Cleanup function
# -------------------------
cleanup() {
    trap - EXIT INT

    echo "[$(date +%H:%M:%S)] Cleaning up..."
    
    # Kill any running main.opt.exe instances
    echo "Killing any running main.opt.exe instances..."
    pkill -f "main.opt.exe" || true
    
    # Kill Xvfb
    pkill -f "Xvfb :1" || true
    
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
# Step 3-5: monitor login and ILLIXR
# -------------------------
found_login=false
found_password=false
logged_in=false

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
        echo "[$TIMESTAMP] Logged in as root → launching ILLIXR"
        screen -S "$SESSION" -X stuff "cd ILLIXR/build\n"
        sleep 1  # Give time for command to execute
        screen -S "$SESSION" -X stuff "export ILLIXR_EYE_TRACKING=2\n"
        sleep 1
        screen -S "$SESSION" -X stuff "./main.opt.exe --yaml=/root/ILLIXR/illixr.yaml\n"
        logged_in=true
    fi

    # Debug: Show current login state
    echo "[$TIMESTAMP] Login state: found_login=$found_login, found_password=$found_password, logged_in=$logged_in"

    # Count openvins once logged in
    if [ "$logged_in" = true ]; then
        # Debug the variables and file
        echo "[$TIMESTAMP] DEBUG: OPENVINS_PREFIX variable: '$OPENVINS_PREFIX'"
        echo "[$TIMESTAMP] DEBUG: Checking file: $TMPFILE"
        echo "[$TIMESTAMP] DEBUG: File size: $(wc -c < "$TMPFILE") bytes"
        
        # Count total occurrences of "openvins:" in the buffer
        openvins_count=$(grep -c "$OPENVINS_PREFIX" "$TMPFILE")
        
        # Also try hardcoded version for comparison
        hardcoded_count=$(grep -c "openvins:" "$TMPFILE")
        
        echo "[$TIMESTAMP] Found $openvins_count '$OPENVINS_PREFIX' occurrences using variable (need $OPENVINS_COUNT)"
        echo "[$TIMESTAMP] Found $hardcoded_count 'openvins:' occurrences using hardcoded string"
        
        if [ "$openvins_count" -ge "$OPENVINS_COUNT" ] || [ "$hardcoded_count" -ge "$OPENVINS_COUNT" ]; then
            # Use whichever count worked
            actual_count=$openvins_count
            if [ "$hardcoded_count" -ge "$OPENVINS_COUNT" ]; then
                actual_count=$hardcoded_count
            fi
            
            echo "[$TIMESTAMP] SUCCESS! Found $actual_count 'openvins:' occurrences. Saving final buffer..."
            
            # Save the final buffer
            cp "$TMPFILE" "$OUTDIR/et-scalar.txt"
            echo "[$TIMESTAMP] Final buffer saved to et-scalar.txt"
            
            echo "Target of $OPENVINS_COUNT openvins occurrences reached! Cleanup will kill FireSim."
            cleanup
        fi
    else
        # Debug: Show why we're not checking for openvins yet
        echo "[$TIMESTAMP] Not logged in yet, skipping openvins count"
    fi

    # Clean up old temp files to save space (keep only last 5)
    ls -t "$OUTDIR"/dump_*.txt 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null

    sleep $INTERVAL
done