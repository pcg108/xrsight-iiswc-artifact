import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from io import StringIO
import sys
import re
from typing import Dict, Any

def parse_screen_buffer(screen_buffer_text, expected_cols=16):
    """
    Parse ov_msckf_timing.txt output from a screen buffer into a DataFrame.
    Assumes `expected_cols` total columns (default 16).
    """
    lines = screen_buffer_text.strip().splitlines()

    # Find where "cat ov_msckf_timing.txt" appears
    try:
        start_idx = next(i for i, l in enumerate(lines) if 'cat ov_msckf_timing.txt' in l)
    except StopIteration:
        raise ValueError("Could not find 'cat ov_msckf_timing.txt' in buffer")

    # Take only the relevant lines
    data_lines = [
        l.strip() for l in lines[start_idx+1:]
        if l.strip() and not l.startswith("root@ubuntu")
    ]

    # ---- Rebuild header ----
    header = ""
    while data_lines and header.count(",") + 1 < expected_cols:
        header += data_lines.pop(0).lstrip("#").strip()
    header = header.replace(" ", "")  # remove spaces in column names like "msckf update" → "msckfupdate"

    # ---- Rebuild rows ----
    rows, buf = [], ""
    for line in data_lines:
        buf += line
        if buf.count(",") + 1 == expected_cols:
            rows.append(buf)
            buf = ""
    if buf:
        rows.append(buf)

    # ---- Build CSV string and parse ----
    csv_text = header + "\n" + "\n".join(rows)
    return csv_text


def plot_timing_data(csv_data):
    """
    Create a stacked bar chart from the timing data
    """
    # Read the CSV data
    df = pd.read_csv(StringIO(csv_data))
    
    # Extract timing columns (up to 'total')
    timing_columns = ['tracking', 'propagation', 'msckfupdate', 'slamupdate', 
                     'slamdelayed', 'marginalization']
    
    # Convert nanoseconds to milliseconds for better readability
    timing_data = df[timing_columns] / 1e6  # Convert ns to ms
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create stacked bar chart
    x = np.arange(len(timing_data))
    width = 0.8
    
    # Colors for each timing component
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
    
    bottom = np.zeros(len(timing_data))
    bars = []
    
    for i, col in enumerate(timing_columns):
        bars.append(ax.bar(x, timing_data[col], width, bottom=bottom, 
                          label=col, color=colors[i % len(colors)]))
        bottom += timing_data[col]
    
    # Customize the plot
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Time (ms)')
    ax.set_title('OpenVINS MSCKF Timing Breakdown (Stacked Bar Chart)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add total time as text on top of each bar
    totals = timing_data.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals) * 0.01, f'{total:.1f}', 
                ha='center', va='bottom', fontsize=8)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show statistics
    print("Timing Statistics (ms):")
    print("=" * 50)
    stats = timing_data.describe()
    print(stats)
    
    print(f"\nTotal samples: {len(timing_data)}")
    print(f"Average total time: {totals.mean():.2f} ms")
    print(f"Max total time: {totals.max():.2f} ms")
    print(f"Min total time: {totals.min():.2f} ms")
    
    return fig, timing_data


def plot_ov_speedup_over_scalar(csv_data_saturn, csv_data_scalar):
    """
    Create a stacked bar chart from the timing data
    """
    # Read the CSV data
    df_saturn = pd.read_csv(StringIO(csv_data_saturn))
    df_scalar = pd.read_csv(StringIO(csv_data_scalar))

    speedup = (df_scalar['total'] - df_saturn['total']) / df_scalar['total']
    speedup = speedup.round(2)

    # Compute 9-frame moving average
    moving_avg = speedup.rolling(window=9, center=True).mean()

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(speedup, label='Speedup (Saturn over Scalar)', color='blue', alpha=0.6)
    plt.plot(moving_avg, label='9-frame Moving Average', color='red', linewidth=2)
    plt.xlabel('Sample / Index')
    plt.ylabel('Speedup')
    plt.title('Saturn vs Scalar Speedup with 9-frame Moving Average')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/ov_speedup_over_scalar.png')

def read_screen_buffer_from_file(filename):
    """
    Read screen buffer data from a file
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file '{filename}': {e}")
        return None


def parse_rvv_test_results(text, test):
    """
    Parse RISC-V vector test results from screen buffer text.
    
    Returns a dictionary with keys:
    - GEMM: Matrix multiplication results (type 3 commands)
    - GEMV_N: Normal matrix-vector multiplication results (type 1 commands)
    - GEMV_T: Transposed matrix-vector multiplication results (type 1 commands)
    
    Each key contains a dictionary mapping matrix/vector sizes to average performance values.
    """
    results = {
        'GEMM': {},
        'GEMV_N': {},
        'GEMV_T': {}
    }
    
    lines = text.split('\n')
    i = 0

    cmd_prefix = f'./rvvTest-{test}'
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for rvvTest commands
        if cmd_prefix in line:
            # Parse command line
            cmd_match = re.search(rf'{re.escape(cmd_prefix)}\s+(\d+)\s+(\d+)', line)
            if cmd_match:
                test_type = int(cmd_match.group(1))
                size = int(cmd_match.group(2))
                
                if test_type == 3:
                    # Matrix multiplication test - look for overall Average
                    j = i + 1
                    while j < len(lines) and j < i + 50:  # Look ahead reasonable distance
                        if 'Average:' in lines[j]:
                            avg_match = re.search(r'Average:\s*(\d+)', lines[j])
                            if avg_match:
                                results['GEMM'][size] = int(avg_match.group(1))
                                break
                        j += 1
                
                elif test_type == 1:
                    # Vector operations test - look for gemv_n and gemv_t averages
                    j = i + 1
                    gemv_n_found = False
                    gemv_t_found = False
                    
                    while j < len(lines) and j < i + 50:  # Look ahead reasonable distance
                        if 'gemv_n(A, X, Y, alpha, beta);' in lines[j]:
                            # Look for the Average line after gemv_n
                            k = j + 1
                            while k < len(lines) and k < j + 5:
                                if 'Average:' in lines[k]:
                                    avg_match = re.search(r'Average:\s*(\d+)', lines[k])
                                    if avg_match:
                                        results['GEMV_N'][size] = int(avg_match.group(1))
                                        gemv_n_found = True
                                        break
                                k += 1
                        
                        elif 'gemv_t(A, X, Y, alpha, beta);' in lines[j]:
                            # Look for the Average line after gemv_t
                            k = j + 1
                            while k < len(lines) and k < j + 5:
                                if 'Average:' in lines[k]:
                                    avg_match = re.search(r'Average:\s*(\d+)', lines[k])
                                    if avg_match:
                                        results['GEMV_T'][size] = int(avg_match.group(1))
                                        gemv_t_found = True
                                        break
                                k += 1
                        
                        # Break if we found both gemv results
                        if gemv_n_found and gemv_t_found:
                            break
                        j += 1
        
        i += 1
    
    return results

def plot_kernel_speedups(sb_saturn, sb_gemmini, sb_generic):
    saturn_kernel = parse_rvv_test_results(sb_saturn, 'zvl256')
    gemmini_kernel = parse_rvv_test_results(sb_gemmini, 'gemmini')
    generic_kernel = parse_rvv_test_results(sb_generic, 'generic')

    dims = sorted(generic_kernel['GEMM'].keys())
    saturn_percent = [(generic_kernel['GEMM'][d] - saturn_kernel['GEMM'][d]) / generic_kernel['GEMM'][d] * 100 for d in dims]
    gemmini_percent = [(generic_kernel['GEMM'][d] - gemmini_kernel['GEMM'][d]) / generic_kernel['GEMM'][d] * 100 for d in dims]
    plt.figure(figsize=(8,5))
    plt.plot(dims, gemmini_percent, 'r-o', label='Gemmini')
    plt.plot(dims, saturn_percent, 'g-o', label='Saturn')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('% Increase over Generic')
    plt.title('GEMM % Increase over Generic Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/gemm_kernel_speedup.png')

    dims = sorted(generic_kernel['GEMV_N'].keys())
    saturn_percent = [(generic_kernel['GEMV_N'][d] - saturn_kernel['GEMV_N'][d]) / generic_kernel['GEMV_N'][d] * 100 for d in dims]
    gemmini_percent = [(generic_kernel['GEMV_N'][d] - gemmini_kernel['GEMV_N'][d]) / generic_kernel['GEMV_N'][d] * 100 for d in dims]
    plt.figure(figsize=(8,5))
    plt.plot(dims, gemmini_percent, 'r-o', label='Gemmini')
    plt.plot(dims, saturn_percent, 'g-o', label='Saturn')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('% Increase over Generic')
    plt.title('GEMV_N % Increase over Generic Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/gemv_n_kernel_speedup.png')

    dims = sorted(generic_kernel['GEMV_T'].keys())
    saturn_percent = [(generic_kernel['GEMV_T'][d] - saturn_kernel['GEMV_T'][d]) / generic_kernel['GEMV_T'][d] * 100 for d in dims]
    gemmini_percent = [(generic_kernel['GEMV_T'][d] - gemmini_kernel['GEMV_T'][d]) / generic_kernel['GEMV_T'][d] * 100 for d in dims]
    plt.figure(figsize=(8,5))
    plt.plot(dims, gemmini_percent, 'r-o', label='Gemmini')
    plt.plot(dims, saturn_percent, 'g-o', label='Saturn')
    plt.xlabel('Matrix Dimension')
    plt.ylabel('% Increase over Generic')
    plt.title('GEMV_T % Increase over Generic Kernel')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/gemv_t_kernel_speedup.png')

# sb_saturn = read_screen_buffer_from_file('screen_dumps/vio_saturn.txt')
# sb_gemmini = read_screen_buffer_from_file('screen_dumps/vio_gemmini.txt')
# sb_generic = read_screen_buffer_from_file('screen_dumps/vio_generic.txt')

# # plot individual kernel speedup (Figure 12)
# plot_kernel_speedups(sb_saturn, sb_generic, sb_generic)

# # plot saturn bars (figure 13a)
# csv_data_saturn = parse_screen_buffer(sb_saturn)
# fig, timing_data = plot_timing_data(csv_data_saturn)            
# output_filename = "figures/saturn_bars.png"
# fig.savefig(output_filename, dpi=300, bbox_inches='tight')
# print(f"Plot saved as: {output_filename}")

# # plot gemmini bars (figure 13b)
# csv_data_gemmini = parse_screen_buffer(sb_gemmini)
# fig, timing_data = plot_timing_data(csv_data_gemmini)            
# output_filename = "figures/gemmini_bars.png"
# fig.savefig(output_filename, dpi=300, bbox_inches='tight')
# print(f"Plot saved as: {output_filename}")
        
# # plot saturn vs scalar for VIO (figure 14)
# csv_data_scalar = parse_screen_buffer(sb_generic)
# plot_ov_speedup_over_scalar(csv_data_saturn, csv_data_scalar)
