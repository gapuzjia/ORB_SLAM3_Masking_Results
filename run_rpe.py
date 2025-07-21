import os
import re
import subprocess

# Ground truth paths
GT_FILES = {
    "MH01": "./EuRoCGroundTruth/GT_MH01.csv",
    "MH02": "./EuRoCGroundTruth/GT_MH02.csv",
    "MH03": "./EuRoCGroundTruth/GT_MH03.csv",
    "MH04": "./EuRoCGroundTruth/GT_MH04.csv",
    "MH05": "./EuRoCGroundTruth/GT_MH05.csv",
    "V101": "./EuRoCGroundTruth/GT_V101.csv",
    "V102": "./EuRoCGroundTruth/GT_V102.csv",
    "V103": "./EuRoCGroundTruth/GT_V103.csv",
    "V201": "./EuRoCGroundTruth/GT_V201.csv",
    "V202": "./EuRoCGroundTruth/GT_V202.csv",
    "V203": "./EuRoCGroundTruth/GT_V203.csv"
}

# Output CSV
rpe_csv = "RPE_results.csv"

# Initialize CSV header
with open(rpe_csv, "w") as f:
    f.write("run_id,dataset,trans_rmse,rot_rmse\n")

# Scan result folders inside MaskingResults/
results_dir = "MaskingResults"

for folder_name in os.listdir(results_dir):
    if not folder_name.startswith("2025"):
        continue

    folder_path = os.path.join(results_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing: {folder_name}")

    # Extract dataset name
    match = re.search(r"(MH[0-9]+|V[0-9]+)", folder_name)
    if not match:
        print(f"Could not extract dataset from {folder_name}, skipping...")
        continue

    dataset = match.group(1)

    if dataset not in GT_FILES:
        print(f"Unknown dataset '{dataset}', skipping...")
        continue

    gt_file = GT_FILES[dataset]
    if not os.path.exists(gt_file):
        print(f"Ground truth file missing: {gt_file}")
        continue

    # Locate keyframe file
    kf_file = os.path.join(folder_path, f"kf_dataset-{dataset}_stereo_imu.txt")
    if not os.path.exists(kf_file):
        print(f"Missing keyframe file in {folder_name}")
        continue

    print(f"Running RPE for dataset {dataset}")
    subprocess.run([
        "python", "scripts/evaluate_rpe_scale.py",
        gt_file, kf_file, "--csv", rpe_csv
    ], check=True)

print("\nAll RPE evaluations complete.")
