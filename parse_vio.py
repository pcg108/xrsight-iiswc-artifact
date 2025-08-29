import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from io import StringIO
import sys
import re
from typing import Dict, Any

def parse_screen_buffer(screen_buffer_text):
    """
    Parse screen buffer text to extract the timing data from ov_msckf_timing.txt
    """
    lines = screen_buffer_text.strip().split('\n')
    
    # Find the line that contains the cat command
    cat_line_idx = -1
    for i, line in enumerate(lines):
        if 'cat ov_msckf_timing.txt' in line:
            cat_line_idx = i
            break
    
    if cat_line_idx == -1:
        raise ValueError("Could not find 'cat ov_msckf_timing.txt' command in buffer")
    
    # Extract data lines (everything after the cat command)
    data_lines = lines[cat_line_idx + 1:]
    
    # Filter out empty lines and command prompts
    clean_data_lines = []
    for line in data_lines:
        line = line.strip()
        if line and not line.startswith('root@ubuntu') and not line.startswith('#'):
            clean_data_lines.append(line)
    
    # The header line should be the first line that starts with #
    header_line = None
    for line in lines[cat_line_idx:]:
        if line.strip().startswith('#'):
            header_line = line.strip()[1:]  # Remove the # symbol
            break
    
    if not header_line:
        # Default header based on the expected format
        header_line = "timestamp (ms),tracking,propagation,msckf update,slam update,slam delayed,marginalization,total,q_GtoI.x,q_GtoI.y,q_GtoI.z,q_GtoI.w,p_IinG.x,p_IinG.y,p_IinG.z,dist"
    
    # Create CSV-like string
    csv_data = header_line + '\n' + '\n'.join(clean_data_lines)
    
    return csv_data

def plot_timing_data(csv_data):
    """
    Create a stacked bar chart from the timing data
    """
    # Read the CSV data
    df = pd.read_csv(StringIO(csv_data))
    
    # Extract timing columns (up to 'total')
    timing_columns = ['tracking', 'propagation', 'msckf update', 'slam update', 
                     'slam delayed', 'marginalization']
    
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

def parse_rvv_test_results(text: str) -> Dict[str, Dict[str, int]]:
    """
    Parse RISC-V test results and extract Average values.
    
    Returns a dictionary with structure:
    {
        'GEMM': {matrix_size: average_value, ...},
        'GEMV_N': {vector_size: average_value, ...},
        'GEMV_T': {vector_size: average_value, ...}
    }
    """
    results = {
        'GEMM': {},
        'GEMV_N': {},
        'GEMV_T': {}
    }
    
    # Split text into sections by command invocations
    command_sections = re.split(r'root@ubuntu:~# (\.\/rvvTest-[^\s]+)', text)
    
    # Process each section (skip first empty element)
    for i in range(1, len(command_sections), 2):
        if i + 1 >= len(command_sections):
            break
            
        command = command_sections[i]
        output = command_sections[i + 1]
        
        # Parse command arguments
        cmd_parts = command.split()
        if len(cmd_parts) < 3:
            continue
            
        # Extract the test type (first argument after executable)
        test_type = cmd_parts[1]
        
        # Extract the size parameter (second argument)
        if len(cmd_parts) >= 3:
            try:
                size_param = int(cmd_parts[2])
            except ValueError:
                continue
        else:
            continue
        
        if test_type == '3':
            # GEMM test - look for single Average
            avg_match = re.search(r'Average:\s*(\d+)', output)
            if avg_match:
                average = int(avg_match.group(1))
                results['GEMM'][size_param] = average
                
        elif test_type == '1':
            # GEMV test - look for gemv_n and gemv_t averages
            
            # Find gemv_n section and its average
            gemv_n_match = re.search(r'gemv_n\([^)]+\);[^A]*?Average:\s*(\d+)', output, re.DOTALL)
            if gemv_n_match:
                average = int(gemv_n_match.group(1))
                results['GEMV_N'][size_param] = average
            
            # Find gemv_t section and its average
            gemv_t_match = re.search(r'gemv_t\([^)]+\);[^A]*?Average:\s*(\d+)', output, re.DOTALL)
            if gemv_t_match:
                average = int(gemv_t_match.group(1))
                results['GEMV_T'][size_param] = average
    
    return results
    
filename = 'screen_dumps/vio_saturn.txt'

print(f"Reading screen buffer from: {filename}")

# Read screen buffer from file
screen_buffer = read_screen_buffer_from_file(filename)

if screen_buffer is not None:
    try:
        csv_data = parse_screen_buffer(screen_buffer)
        fig, timing_data = plot_timing_data(csv_data)            

        output_filename = "figures/saturn_bars.png"
        fig.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_filename}")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        print("Please check your screen buffer format")
else:
    print("Failed to read screen buffer file. Please check the file path and try again.")



