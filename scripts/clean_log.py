#!/usr/bin/env python3
"""
Clean progress bar output from log files.
Removes lines containing carriage returns (progress bar updates).

Usage:
    python scripts/clean_log.py <input_log> [output_log]
    python scripts/clean_log.py grid_search_log.0 grid_search_clean.log
"""

import sys
import re
import os


def clean_log(input_path, output_path=None):
    """
    Remove progress bar lines from a log file.
    Progress bars are characterized by:
    - Lines with \r (carriage return) without \n
    - Lines matching patterns like "123/456 [====..."
    - Lines with ETA: or just time remaining
    - Truncated/incomplete lines (carriage return artifacts)
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_clean{ext}"

    # Patterns to filter out (aggressive)
    progress_patterns = [
        r'^\s*\d+/\d+\s*\[',           # "  99/829 [==>..."
        r'^\s*\d+/\d+\s+━',             # Unicode progress bars
        r'ETA:\s*\d+',                   # "ETA: 34s"
        r'^\s*$',                        # Empty lines
        r'\[=+>?\.*\]',                  # [=====>....] progress bars
        r'- loss:.*accuracy:',           # Training progress updates
        r'^\s*\d+\s*samples',            # Sample counts
        r'it/s\]',                        # tqdm iteration speed
        r'^\s*(100)?%\|',                # Percentage bars
        r'\d+:\d+<',                      # Time remaining
        r'━+',                            # Solid progress bars
        r'^\s*\d+\]',                     # Truncated " 2]" (carriage return artifact)
        r'^\[Epoch\s*$',                  # Incomplete "[Epoch" without rest
    ]
    progress_regex = re.compile('|'.join(progress_patterns))

    # Count stats
    total_lines = 0
    filtered_lines = 0
    kept_lines = 0

    with open(input_path, 'rb') as f_in:
        raw_content = f_in.read()

    # Remove carriage returns and backspaces (control characters that cause overwrites)
    raw_content = raw_content.replace(b'\r', b'')      # Carriage return (line rewind)
    raw_content = raw_content.replace(b'\b', b'')      # Backspace character
    raw_content = raw_content.replace(b'\x08', b'')    # Alternative backspace (same as \b)

    # Decode and process
    content = raw_content.decode('utf-8', errors='replace')

    with open(output_path, 'w') as f_out:
        for line in content.split('\n'):
            total_lines += 1

            # Skip progress bar lines
            if progress_regex.search(line):
                filtered_lines += 1
                continue

            # Skip lines that are just whitespace
            if not line.strip():
                filtered_lines += 1
                continue

            f_out.write(line + '\n')
            kept_lines += 1

    # Print summary
    print(f"Log cleaning complete:")
    print(f"  Input:  {input_path} ({os.path.getsize(input_path) / 1024 / 1024:.1f} MB)")
    print(f"  Output: {output_path} ({os.path.getsize(output_path) / 1024:.1f} KB)")
    print(f"  Lines: {total_lines:,} total → {kept_lines:,} kept ({filtered_lines:,} filtered)")
    print(f"  Reduction: {(1 - os.path.getsize(output_path) / os.path.getsize(input_path)) * 100:.1f}%")

    return output_path


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        sys.exit(1)
    
    clean_log(input_path, output_path)


if __name__ == "__main__":
    main()
