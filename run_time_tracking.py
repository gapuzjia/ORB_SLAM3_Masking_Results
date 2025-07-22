import os
import csv

results_dir = "MaskingResults"
output_csv = "TrackingTimeAvg.csv"

# Create header
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["run_id", "avg_Total_ms"])

# Process each result folder
for folder in os.listdir(results_dir):
    if not folder.startswith("2025"):
        continue

    folder_path = os.path.join(results_dir, folder)
    tracking_file = os.path.join(folder_path, "TrackingTimeStats.txt")

    if not os.path.isfile(tracking_file):
        print(f"[WARN] TrackingTimeStats.txt not found in {folder}")
        continue

    with open(tracking_file, "r") as f:
        header = f.readline().strip().split(",")
        try:
            total_index = header.index("Total[ms]")
        except ValueError:
            print(f"[WARN] 'Total[ms]' column not found in {tracking_file}")
            continue

        total_values = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= total_index:
                continue
            try:
                total_values.append(float(parts[total_index]))
            except ValueError:
                continue

        if not total_values:
            print(f"[WARN] No valid data in {tracking_file}")
            continue

        avg_total = sum(total_values) / len(total_values)

    # Write to CSV
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([folder, f"{avg_total:.6f}"])

    print(f"[✓] {folder}: avg_Total = {avg_total:.3f} ms")

print("\nAll averages written to", output_csv)
