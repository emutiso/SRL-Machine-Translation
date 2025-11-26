"""
Script to convert human evaluation text file to CSV format
"""
import os
import re
import csv
import argparse
from pathlib import Path

def convert_evaluation_to_csv(input_file, output_file):
    """
    Convert human evaluation text file to CSV format

    Args:
        input_file: Path to input text file
        output_file: Path to output CSV file
    """
    print(f"Converting {input_file} to {output_file}...")

    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split the content into samples
    # Each sample starts with "## Sample X"
    sample_blocks = re.split(r'## Sample \d+', content)[1:]  # Skip header
    sample_numbers = re.findall(r'## Sample (\d+)', content)

    # Create CSV rows
    rows = []
    headers = ["Sample", "Source", "Reference", "Baseline_Translation",
               "SRL_Augmented_Translation", "Baseline_Score", "SRL_Score", "Comments"]

    for i, (sample_num, block) in enumerate(zip(sample_numbers, sample_blocks)):
        # Extract fields using regex
        source_match = re.search(r'Source: (.*?)(?=\nSRL-Tagged:|$)', block, re.DOTALL)
        reference_match = re.search(r'Reference: (.*?)(?=\n\nBaseline|$)', block, re.DOTALL)
        baseline_match = re.search(r'Baseline Translation: (.*?)(?=\nSRL-Augmented|$)', block, re.DOTALL)
        srl_match = re.search(r'SRL-Augmented Translation: (.*?)(?=\nHuman Rating|$)', block, re.DOTALL)

        # Extract the text and clean it
        source = source_match.group(1).strip() if source_match else ""
        reference = reference_match.group(1).strip() if reference_match else ""
        baseline = baseline_match.group(1).strip() if baseline_match else ""
        srl = srl_match.group(1).strip() if srl_match else ""

        # Create row
        row = {
            "Sample": sample_num,
            "Source": source,
            "Reference": reference,
            "Baseline_Translation": baseline,
            "SRL_Augmented_Translation": srl,
            "Baseline_Score": "",
            "SRL_Score": "",
            "Comments": ""
        }

        rows.append(row)

    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully converted {len(rows)} samples to CSV format")
    return rows

def compare_csv_files(file1, file2):
    """
    Compare two CSV files to check for differences

    Args:
        file1: Path to first CSV file
        file2: Path to second CSV file
    """
    print(f"Comparing {file1} and {file2}...")

    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        reader1 = csv.DictReader(f1)
        reader2 = csv.DictReader(f2)

        rows1 = list(reader1)
        rows2 = list(reader2)

        if len(rows1) != len(rows2):
            print(f"Row count mismatch: {file1} has {len(rows1)} rows, {file2} has {len(rows2)} rows")
            return False

        # Important columns to check
        columns = ["Source", "Reference", "Baseline_Translation", "SRL_Augmented_Translation"]

        for i, (row1, row2) in enumerate(zip(rows1, rows2)):
            for column in columns:
                if row1[column] != row2[column]:
                    print(f"Difference in sample {i+1}, column {column}:")
                    print(f"  {file1}: {row1[column][:50]}...")
                    print(f"  {file2}: {row2[column][:50]}...")
                    return False

    print("CSV files are identical for source texts and translations!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert human evaluation text file to CSV format")
    parser.add_argument("input_file", nargs='?', default="human_evaluation_samples_20250403_115601.txt",
                        help="Path to input text file")
    parser.add_argument("--output", "-o", default="human_evaluation_script_generated.csv",
                        help="Path to output CSV file")
    parser.add_argument("--compare", "-c", default="human_evaluation_20250403_115601.csv",
                        help="Path to CSV file to compare with (for validation)")

    args = parser.parse_args()

    convert_evaluation_to_csv(args.input_file, args.output)

    if args.compare and os.path.exists(args.compare):
        compare_csv_files(args.output, args.compare)