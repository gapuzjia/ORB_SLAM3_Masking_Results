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
ate_csv = "ATE_results.csv"
rpe_csv = "RPE_results.csv"
plot_dir = "ResultsErrorOverTime"
os.makedirs(plot_dir, exist_ok=True)

# Init CSV headers
with open(ate_csv, "w") as f:
    f.write("run_id,rmse,mean,max,std,median\n")

with open(rpe_csv, "w") as f:
    f.write("run_id,dataset,trans_rmse,rot_rmse\n")

# Search MaskingResults for folders starting with 2025
for folder_name in os.listdir("MaskingResults"):
    if not folder_name.startswith("2025"):
        continue

    folder_path = os.path.join("MaskingResults", folder_name)
    print(f"\nChecking folder: {folder_name}")

    # Identify dataset from folder name
    dataset = None
    for key in GT_FILES:
        if key in folder_name:
            dataset = key
            break

    if dataset is None:
        print(f"[WARN] Could not determine dataset for: {folder_name}")
        continue

    gt_file = GT_FILES[dataset]
    if not os.path.isfile(gt_file):
        print(f"[WARN] Missing ground truth file: {gt_file}")
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
        print(f"[WARN] Missing f/kf files in {folder_name}")
        continue

    run_id = folder_name

    # ----------- ATE -----------
    print("Running ATE")
    tmp_csv = "temp_ate.csv"
    subprocess.run([
        "python", "scripts/evaluate_ate_scale.py",
        gt_file, f_file, "--csv_output", tmp_csv
    ])

    if os.path.exists(tmp_csv):
        with open(tmp_csv) as tempf, open(ate_csv, "a") as outf:
            lines = tempf.readlines()[1:]  # skip header
            for line in lines:
                outf.write(f"{run_id},{line.strip()}\n")
        os.remove(tmp_csv)
    else:
        print(f"[WARN] ATE failed for {folder_name}")

    # # ----------- RPE -----------
    # print("Running RPE")
    # subprocess.run([
    # "python", "scripts/evaluate_rpe_scale.py",
    # gt_file, kf_file, "--csv", rpe_csv
    # ], check=True)



    # ----------- ERROR OVER TIME -----------
    print("Generating Error Over Time Plot")
    plot_output = os.path.join(plot_dir, f"{folder_name}_error_plot.svg")
    subprocess.run([
        "python", "scripts/evaluate_error_over_time.py",
        gt_file, f_file, "--plot", plot_output
    ])

    print(f"Done with: {folder_name}")

print("\nAll evaluations complete.")
