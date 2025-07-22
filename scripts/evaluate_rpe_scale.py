import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import os
import csv

def read_trajectory(file_path):
    trajectory = {}
    with open(file_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().replace(',', ' ').split()
            timestamp = float(parts[0])
            position = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            orientation = R.from_quat([
                float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            ]).as_matrix()
            pose = np.eye(4)
            pose[:3, :3] = orientation
            pose[:3, 3] = position
            trajectory[timestamp] = pose
    return trajectory

def compute_rpe(gt_trajectory, est_trajectory):
    timestamps = sorted(set(gt_trajectory.keys()).intersection(est_trajectory.keys()))
    trans_errors = []
    rot_errors = []

    for i in range(len(timestamps) - 1):
        t1, t2 = timestamps[i], timestamps[i + 1]
        gt_rel = np.linalg.inv(gt_trajectory[t1]) @ gt_trajectory[t2]
        est_rel = np.linalg.inv(est_trajectory[t1]) @ est_trajectory[t2]
        error_mat = np.linalg.inv(gt_rel) @ est_rel

        trans_error = np.linalg.norm(error_mat[:3, 3])
        rot_error = R.from_matrix(error_mat[:3, :3]).magnitude()

        trans_errors.append(trans_error)
        rot_errors.append(np.degrees(rot_error))

    return trans_errors, rot_errors

def plot_errors(trans_errors, rot_errors, output_path=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(trans_errors, label="Translational Error (m)")
    ax[1].plot(rot_errors, label="Rotational Error (deg)", color='orange')
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel("Meters")
    ax[1].set_ylabel("Degrees")
    ax[1].set_xlabel("Frame Index")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)

def write_rpe_metrics_to_csv(csv_path, run_id, dataset, trans_errors, rot_errors):
    rmse_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rmse_rot = np.sqrt(np.mean(np.square(rot_errors)))
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["run_id", "dataset", "trans_rmse", "rot_rmse"])
        writer.writerow([run_id, dataset, rmse_trans, rmse_rot])

def parse_run_metadata(file_path):
    filename = os.path.basename(file_path)
    parts = filename.replace(".txt", "").split("-")
    dataset = parts[1] if len(parts) > 1 else "unknown"
    run_id = filename.replace(".txt", "")
    return run_id, dataset, mask

def main():
    parser = argparse.ArgumentParser(description="Compute Relative Pose Error (RPE) with RMSE")
    parser.add_argument("groundtruth_file", help="Ground truth trajectory file")
    parser.add_argument("estimated_file", help="Estimated trajectory file")
    parser.add_argument("--plot", help="Path to save error plot", default=None)
    parser.add_argument("--csv", help="Path to save CSV summary", default=None)
    args = parser.parse_args()

    gt_traj = read_trajectory(args.groundtruth_file)
    est_traj = read_trajectory(args.estimated_file)

    trans_errors, rot_errors = compute_rpe(gt_traj, est_traj)

    rmse_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rmse_rot = np.sqrt(np.mean(np.square(rot_errors)))

    print(f"Translational RMSE: {rmse_trans:.4f} m")
    print(f"Rotational RMSE: {rmse_rot:.4f} deg")

    if args.plot:
        plot_errors(trans_errors, rot_errors, args.plot)
    else:
        plot_errors(trans_errors, rot_errors)

    run_id = os.path.basename(os.path.dirname(args.estimated_file))
    dataset = run_id.split('_')[4]

    if args.csv:
        write_rpe_metrics_to_csv(args.csv, run_id, dataset, trans_errors, rot_errors)

if __name__ == "__main__":
    main()