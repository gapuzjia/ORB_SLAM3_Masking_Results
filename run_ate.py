import os
import re
import subprocess

# Define known datasets and their GT paths
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

# Output files
ate_csv = "Host_StaticATE_results.csv"
rpe_csv = "Host_StaticRPE_results.csv"
plot_dir = "Host_StaticResultsErrorOverTime"
os.makedirs(plot_dir, exist_ok=True)

# Init CSV headers
with open(ate_csv, "w") as f:
    f.write("run_id,dataset,rmse,mean,max,std,median,scale_factor\n")

with open(rpe_csv, "w") as f:
    f.write("run_id,dataset,trans_rmse,trans_mean,trans_median,trans_std,trans_max,rot_rmse,rot_mean,rot_median,rot_std,rot_max\n")

# Gather and sort all valid folders
folders = sorted([
    name for name in os.listdir("MaskingResults")
    if name.startswith("2026") and os.path.isdir(os.path.join("MaskingResults", name))
])

total = len(folders)
print(f"Found {total} folders to process.\n")

for i, folder_name in enumerate(folders):
    folder_path = os.path.join("MaskingResults", folder_name)
    print(f"\n[{i+1}/{total}] Processing: {folder_name}")

    # Identify dataset from folder name
    dataset = None
    for key in GT_FILES:
        if key in folder_name:
            dataset = key
            break

    if dataset is None:
        print(f"  [WARN] Could not determine dataset for: {folder_name}")
        continue

    gt_file = GT_FILES[dataset]
    if not os.path.isfile(gt_file):
        print(f"  [WARN] Missing ground truth file: {gt_file}")
        continue

    # Look for f and kf files
    f_file = None
    kf_file = None
    for file in os.listdir(folder_path):
        if re.match(rf"f_dataset-{dataset}_stereo_imu\.txt", file):
            f_file = os.path.join(folder_path, file)
        elif re.match(rf"kf_dataset-{dataset}_stereo_imu\.txt", file):
            kf_file = os.path.join(folder_path, file)

    if not f_file or not kf_file:
        print(f"  [WARN] Missing f/kf files in {folder_name}")
        if not f_file:
            print(f"         Missing: f_dataset-{dataset}_stereo_imu.txt")
        if not kf_file:
            print(f"         Missing: kf_dataset-{dataset}_stereo_imu.txt")
        continue

    run_id = folder_name

    # ----------- ATE -----------
    print(f"  Running ATE...")
    tmp_csv = "temp_ate.csv"
    result = subprocess.run([
        "python", "scripts/evaluate_ate_scale.py",
        gt_file, f_file, "--csv_output", tmp_csv
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [WARN] ATE failed for {folder_name}")
        print(f"         stdout: {result.stdout.strip()}")
        print(f"         stderr: {result.stderr.strip()}")
    elif os.path.exists(tmp_csv):
        with open(tmp_csv) as tempf, open(ate_csv, "a") as outf:
            lines = tempf.readlines()[1:]  # skip header
            for line in lines:
                outf.write(f"{run_id},{line.strip()}\n")
        os.remove(tmp_csv)
        print(f"  ATE results saved.")
    else:
        print(f"  [WARN] ATE returned success but temp CSV not found for {folder_name}")

    # ----------- RPE -----------
    print(f"  Running RPE...")
    result = subprocess.run([
        "python", "scripts/evaluate_rpe_scale.py",
        gt_file, kf_file, "--csv", rpe_csv
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [WARN] RPE failed for {folder_name}")
        print(f"         stdout: {result.stdout.strip()}")
        print(f"         stderr: {result.stderr.strip()}")
    else:
        print(f"  RPE results saved.")

    # ----------- ERROR OVER TIME -----------
    print(f"  Generating Error Over Time plot...")
    plot_output = os.path.join(plot_dir, f"{folder_name}_error_plot.svg")
    result = subprocess.run([
        "python", "scripts/evaluate_error_over_time.py",
        gt_file, f_file, "--plot", plot_output
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [WARN] Plot failed for {folder_name}")
        print(f"         stdout: {result.stdout.strip()}")
        print(f"         stderr: {result.stderr.strip()}")
    elif os.path.exists(plot_output):
        print(f"  Plot saved: {plot_output}")
    else:
        print(f"  [WARN] Plot returned success but SVG not found for {folder_name}")

    print(f"  Done with: {folder_name}")

print("\nAll evaluations complete.")
print(f"ATE results: {ate_csv}")
print(f"RPE results: {rpe_csv}")
print(f"Plots: {plot_dir}/")