import re
import matplotlib.pyplot as plt
import pickle
import numpy as np

def parse_data_file(filename, sel=[0,1,3,4,5,6,7,9,13,14]):
    result = {}
    current_key = None
    buffer = ""
    number_pattern = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

    def try_extract_numbers(text):
        # Extract numbers using regex and return as floats
        matches = number_pattern.findall(text)
        return [float(m) for m in matches]

    with open(filename, 'r') as f:

        start_reading = False

        for line in f:
            line = line.strip()

            if not line:
                continue  # skip empty lines

            if not start_reading:
                if 'waiting for enough clone states (4 of 5)....' not in line:
                    continue  # skip until we find the start marker
                start_reading = True
                continue

            if ':' in line:
                # New key starts
                if current_key and buffer:
                    nums = try_extract_numbers(buffer)
                    if len(nums) == 29:
                        selected = [nums[i] for i in sel]
                        result.setdefault(current_key, []).append(selected)
                # Split the line and start a new buffer
                parts = line.split(':', 1)
                current_key = parts[0].strip()
                buffer = parts[1].strip()
            else:
                # Continuation line; append to buffer
                buffer += line.strip()

            # Try extracting numbers after each line
            nums = try_extract_numbers(buffer)
            if len(nums) == 29:
                selected = [nums[i] for i in sel]
                result.setdefault(current_key, []).append(selected)
                buffer = ""
                current_key = None

    # Handle any trailing data at EOF
    if current_key and buffer:
        nums = try_extract_numbers(buffer)
        if len(nums) == 29:
            selected = [nums[i] for i in sel]
            result.setdefault(current_key, []).append(selected)

    if None in result:
        del result[None]

    return result


def gpu_plot(filename):
    samples = []

    def extract_floats(text):
        """Extract valid floats from messy text with potential linebreaks."""
        float_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')
        return [float(m.group()) for m in float_regex.finditer(text)]

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()

            if '[gpu_model]' in line:
                # Extract moving average
                match = re.search(r"moving average: ([\d\.]+)", line)
                moving_avg = float(match.group(1)) if match else None

                # Extract shading rate
                match = re.search(r"shading rate:\s*([^\s,]+)", line)
                shading_rate = match.group(1) if match else None

                # Look for headless_native_renderer
                while True:
                    line = f.readline()
                    if not line:
                        break  # EOF

                    line = line.strip()
                    if line.startswith('headless_native_renderer:'):
                        renderer_prefix = 'headless_native_renderer:'
                        content = line[len(renderer_prefix):].strip()

                        # Keep reading lines and appending until 29 floats are found
                        full_text = content
                        current_numbers = extract_floats(full_text)

                        while len(current_numbers) < 29:
                            next_line = f.readline()
                            if not next_line:
                                break  # EOF
                            next_line = next_line.strip()
                            full_text += next_line  # append raw text, don't add spaces!
                            current_numbers = extract_floats(full_text)

                        # Done — print result
                        # print(f"Moving average: {moving_avg}, Shading rate: {shading_rate}")
                        # print(f"Renderer stats ({len(current_numbers)}): {current_numbers[:29]}")
                        break  # Go back to main loop to look for next [gpu_model]
                
                samples.append([moving_avg, shading_rate, current_numbers[13]])


    samples = samples[:200]


    # Convert string column to float
    data_float = [[row[0] / 1e6, float(row[1]), row[2]] for row in samples]

    # Prepare data
    x = list(range(len(samples)))  # index of sublist
    y1 = [row[0] for row in data_float]  # Render Latency
    y2 = [row[1] for row in data_float]  # Shading Rate
    y3 = [row[2] for row in data_float]  # App Loop Cycles

    # Create subplots: 2 subfigures, shared x-axis
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Top plot with 2 Y-axes ---
    ax1 = axs[0]
    ax2 = ax1.twinx()

    font = 15

    # Left Y-axis: Render Latency
    ax1.plot(x, y1, color='tab:red', label='Render Latency')
    ax1.set_ylabel('Render Latency (ms)', color='tab:red', fontsize=font)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Right Y-axis: Shading Rate
    ax2.plot(x, y2, color='tab:blue', linestyle='--', label='Shading Rate')
    ax2.set_ylabel('Fragment Shading\nRate', color='tab:blue', fontsize=font)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['1:1', '2:1', '4:1'],)

    # # Optional: combine legends
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    # --- Bottom plot: Application Loop Cycles ---
    axs[1].plot(x, y3, color='tab:green')
    axs[1].set_ylabel('App. Loop Cycles', color='tab:green', fontsize=font)
    axs[1].set_xlabel('Frame #', fontsize=font)
    axs[1].tick_params(axis='y', labelcolor='tab:green')

    # Title + save
    plt.suptitle('Adaptive Foveated Rendering Driven by Real-Time GPU Performance', fontsize=font)
    # plt.tight_layout()
    plt.savefig("figures/gpu_case_study.png", bbox_inches='tight', dpi=300)


def gtsam_comparison(gtsam_scalar, gtsam_gemm):

    scalar = condense_list_of_lists(gtsam_scalar['gtsam_integrator'][:400], factor=10)
    gemm = condense_list_of_lists(gtsam_gemm['gtsam_integrator'][:400], factor=10)

    scalar_cycles = [row[8] for row in scalar]
    gemm_cycles = [row[8] for row in gemm]

    scalar_cache_misses = [row[0] + row[1] for row in scalar]
    gemm_cache_misses = [row[0] + row[1] for row in gemm]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Primary y-axis: Cycles
    ax1.plot(scalar_cycles, label='Single Cycles', color='blue', linestyle='-')
    ax1.plot(gemm_cycles, label='Co-located Cycles', color='orange', linestyle='-')
    ax1.set_xlabel("Time Step", fontsize=20)
    ax1.set_ylabel("Cycles", color='black', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black')

    # Secondary y-axis: Cache Misses
    ax2 = ax1.twinx()
    ax2.plot(scalar_cache_misses, label='Single Cache Misses', color='blue', linestyle='--')
    ax2.plot(gemm_cache_misses, label='Co-located Cache Misses', color='orange', linestyle='--')
    ax2.set_ylabel("Cache Misses", color='black', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black')

    # Combined Legend (inside plot area)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='upper right', bbox_to_anchor=(0.98, 0.98),
               fontsize='small', frameon=True)

    plt.title("GTSAM Scalar Core vs Gemmini: Cycles and Cache Misses", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/gtsam_comparison.png", bbox_inches='tight', dpi=300)


def vio_comparison(gtsam_scalar, gtsam_gemm):

    scalar = gtsam_scalar['openvins'][:80]
    gemm = gtsam_gemm['openvins'][:80]

    scalar_cycles = [row[8] for row in scalar]
    gemm_cycles = [row[8] for row in gemm]

    scalar_cache_misses = [row[0] + row[1] for row in scalar]
    gemm_cache_misses = [row[0] + row[1] for row in gemm]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Primary y-axis: Cycles
    ax1.plot(scalar_cycles, label='Single Cycles', color='blue', linestyle='-')
    ax1.plot(gemm_cycles, label='Co-located Cycles', color='orange', linestyle='-')
    ax1.set_xlabel("Time Step", fontsize=20)
    ax1.set_ylabel("Cycles", color='black', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black')

    # Secondary y-axis: Cache Misses
    ax2 = ax1.twinx()
    ax2.plot(scalar_cache_misses, label='Single Cache Misses', color='blue', linestyle='--')
    ax2.plot(gemm_cache_misses, label='Co-located Cache Misses', color='orange', linestyle='--')
    ax2.set_ylabel("Cache Misses", color='black', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='black')

    # Combined Legend (inside plot area)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2,
               loc='upper right', bbox_to_anchor=(0.98, 0.98),
               fontsize='small', frameon=True)

    plt.title("VIO Co-located with IMU Integration", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/ov_comparison.png", bbox_inches='tight', dpi=300)


def condense_list_of_lists(data, factor=10):
    data = np.array(data)
    n = len(data)

    # Shorten to nearest multiple of factor
    new_len = (n // factor) * factor
    if new_len != n:
        data = data[:new_len]
        n = new_len

    reshaped = data.reshape(n // factor, factor, -1)  # (num_groups, factor, sublist_len)
    averaged = reshaped.mean(axis=1)                  # average over factor dimension

    return averaged.tolist()

def plot_counters(data):
    plugins = ['pose_prediction', 'gtsam_integrator', 'openvins', 'eye_tracking_target'] 

    data['gtsam_integrator'] = condense_list_of_lists(data['gtsam_integrator'], factor=10)

    plugin_cycles = {}
    plugin_instructions = {}
    plugin_cache_misses = {}
    plugin_interlocks = {}
    plugin_branch_mispredictions = {}
    plugin_pipeline_flushes = {}

    for plugin in plugins:
        plugin_cycles[plugin] = [row[8] for row in data[plugin]]
        plugin_instructions[plugin] = [row[9] for row in data[plugin]]
        plugin_cache_misses[plugin] = [row[0] + row[1] for row in data[plugin]]
        plugin_interlocks[plugin] = [row[2] + row[3] + row[4] + row[5] for row in data[plugin]]
        plugin_branch_mispredictions[plugin] = [row[6] for row in data[plugin]]
        plugin_pipeline_flushes[plugin] = [row[7] for row in data[plugin]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    shortest = min(len(cycles) for cycles in plugin_cycles.values())

    # Use a single set of handles and labels
    handles = []
    labels = []

    names = {'pose_prediction': 'Pose Prediction',
             'gtsam_integrator': 'IMU Integrator',
             'openvins': 'VIO',
             'eye_tracking_target': 'Gaze Segmentation'}
    for plugin in plugins:
        line, = axes[0, 0].plot(plugin_cycles[plugin][:shortest], label=plugin)
        handles.append(line)
        labels.append(names[plugin])

    axes[0, 0].set_title("Cycles")
    # axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("Cycles")
    axes[0, 0].grid(True)

    for plugin in plugins:
        axes[0, 1].plot(plugin_instructions[plugin][:shortest])
    axes[0, 1].set_title("Instructions Retired")
    # axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Instructions")
    axes[0, 1].grid(True)

    for plugin in plugins:
        axes[0, 2].plot(plugin_cache_misses[plugin][:shortest])
    axes[0, 2].set_title("Cache Misses")
    # axes[0, 2].set_xlabel("Time Step")
    axes[0, 2].set_ylabel("Cache Misses")
    axes[0, 2].grid(True)

    for plugin in plugins:
        axes[1, 0].plot(plugin_interlocks[plugin][:shortest])
    axes[1, 0].set_title("Interlocks")
    axes[1, 0].set_xlabel("Time Step")
    axes[1, 0].set_ylabel("Interlocks")
    axes[1, 0].grid(True)

    for plugin in plugins:
        axes[1, 1].plot(plugin_branch_mispredictions[plugin][:shortest])
    axes[1, 1].set_title("Branch Mispredictions")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Branch Mispredictions")
    axes[1, 1].grid(True)

    for plugin in plugins:
        axes[1, 2].plot(plugin_pipeline_flushes[plugin][:shortest])
    axes[1, 2].set_title("Pipeline Flushes")
    axes[1, 2].set_xlabel("Time Step")
    axes[1, 2].set_ylabel("Pipeline Flushes")
    axes[1, 2].grid(True)

    # Shared legend outside the plot
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout()
    plt.savefig("figures/counters_over_time.png", bbox_inches='tight', dpi=300)


# take the 3 data files and parse them: ov-gemm, ov-gtsam-gemm, and scalar
# grab the eye tracking from scalar and put it in ov-gemm dict
# generate 6 counter plot with ov-gemm
# use ov-gemm and ov-gtsam-gemm to generate the co-location plots
# use ov-gemm to generate the gpu plot

ov_gemm_data = parse_data_file("screen_dumps/ov-gemm.txt")
et_data = parse_data_file("screen_dumps/et-scalar.txt")
ov_gemm_data['eye_tracking_target'] = et_data['eye_tracking_target']

# plot the counters for ov-gemm
plot_counters(ov_gemm_data)

# plot the gpu case study
gpu_plot("screen_dumps/ov-gemm.txt")

ov_gtsam_gemm_data = parse_data_file("screen_dumps/ov-gtsam-gemm.txt")
ov_gemm_data = parse_data_file("screen_dumps/ov-gemm.txt")
# # plot the co-location plots
gtsam_comparison(ov_gemm_data, ov_gtsam_gemm_data)
vio_comparison(ov_gemm_data, ov_gtsam_gemm_data)