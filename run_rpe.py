#!/usr/bin/env python3
# Relative Pose Error (RPE) evaluation for visual-inertial SLAM
#
# Based on the methodology described in:
#   Sturm, J., Engelhard, N., Endres, F., Burgard, W., & Cremers, D. (2012).
#   "A Benchmark for the Evaluation of RGB-D SLAM Systems."
#   In Proc. of the International Conference on Intelligent Robots and Systems (IROS).
#   https://vision.in.tum.de/data/datasets/rgbd-dataset
#
# This implementation computes frame-to-frame relative pose error with proper
# timestamp association for stereo-inertial odometry evaluation.
#
# Software License Agreement (BSD License)
# See LICENSE file for details.

import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')  # For saving plots without display
import matplotlib.pyplot as plt
import os
import csv

def read_trajectory(file_path):
    """Read trajectory file and return dict of timestamp -> pose matrix"""
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

def associate_timestamps(gt_stamps, est_stamps, max_diff=0.02):
    """Associate timestamps from GT and estimated trajectories within max_diff tolerance"""
    matches = []
    gt_sorted = sorted(gt_stamps)
    est_sorted = sorted(est_stamps)
    
    for gt_time in gt_sorted:
        # Find closest estimated timestamp
        best_match = None
        best_diff = float('inf')
        
        for est_time in est_sorted:
            diff = abs(gt_time - est_time)
            if diff < best_diff and diff < max_diff:
                best_diff = diff
                best_match = est_time
        
        if best_match is not None:
            matches.append((gt_time, best_match))
    
    return matches

def compute_rpe(gt_trajectory, est_trajectory, max_time_diff=0.02):
    """Compute Relative Pose Error with proper timestamp association"""
    # Associate timestamps
    matches = associate_timestamps(gt_trajectory.keys(), est_trajectory.keys(), max_time_diff)
    
    if len(matches) < 2:
        raise ValueError(f"Insufficient timestamp matches: {len(matches)}")
    
    trans_errors = []
    rot_errors = []

    for i in range(len(matches) - 1):
        gt_t1, est_t1 = matches[i]
        gt_t2, est_t2 = matches[i + 1]
        
        # Relative motion in ground truth
        gt_rel = np.linalg.inv(gt_trajectory[gt_t1]) @ gt_trajectory[gt_t2]
        # Relative motion in estimate
        est_rel = np.linalg.inv(est_trajectory[est_t1]) @ est_trajectory[est_t2]
        # Error in relative motion
        error_mat = np.linalg.inv(gt_rel) @ est_rel

        trans_error = np.linalg.norm(error_mat[:3, 3])
        rot_error = R.from_matrix(error_mat[:3, :3]).magnitude()

        trans_errors.append(trans_error)
        rot_errors.append(np.degrees(rot_error))

    return trans_errors, rot_errors

def plot_errors(trans_errors, rot_errors, output_path):
    """Plot translational and rotational errors"""
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(trans_errors, label="Translational Error (m)", linewidth=0.8)
    ax[1].plot(rot_errors, label="Rotational Error (deg)", color='orange', linewidth=0.8)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel("Meters")
    ax[1].set_ylabel("Degrees")
    ax[1].set_xlabel("Frame Pair Index")
    ax[0].grid(True, alpha=0.3)
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def write_rpe_metrics_to_csv(csv_path, run_id, dataset, trans_errors, rot_errors):
    """Write RPE RMSE metrics to CSV"""
    rmse_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rmse_rot = np.sqrt(np.mean(np.square(rot_errors)))
    mean_trans = np.mean(trans_errors)
    mean_rot = np.mean(rot_errors)
    median_trans = np.median(trans_errors)
    median_rot = np.median(rot_errors)
    
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["run_id", "dataset", "trans_rmse", "trans_mean", "trans_median", 
                           "rot_rmse", "rot_mean", "rot_median"])
        writer.writerow([run_id, dataset, rmse_trans, mean_trans, median_trans,
                        rmse_rot, mean_rot, median_rot])

def main():
    parser = argparse.ArgumentParser(description="Compute Relative Pose Error (RPE)")
    parser.add_argument("groundtruth_file", help="Ground truth trajectory file")
    parser.add_argument("estimated_file", help="Estimated trajectory file")
    parser.add_argument("--plot", help="Path to save error plot (SVG/PNG)", default=None)
    parser.add_argument("--csv", help="Path to save CSV summary", default=None)
    parser.add_argument("--max_difference", help="Max time difference for association (seconds)", 
                       type=float, default=0.02)
    args = parser.parse_args()

    # Read trajectories
    gt_traj = read_trajectory(args.groundtruth_file)
    est_traj = read_trajectory(args.estimated_file)

    print(f"Ground truth poses: {len(gt_traj)}")
    print(f"Estimated poses: {len(est_traj)}")

    # Compute RPE
    trans_errors, rot_errors = compute_rpe(gt_traj, est_traj, args.max_difference)

    # Calculate statistics
    rmse_trans = np.sqrt(np.mean(np.square(trans_errors)))
    rmse_rot = np.sqrt(np.mean(np.square(rot_errors)))
    mean_trans = np.mean(trans_errors)
    mean_rot = np.mean(rot_errors)

    print(f"\nRPE Results:")
    print(f"  Matched pose pairs: {len(trans_errors)}")
    print(f"  Translational RMSE: {rmse_trans:.4f} m")
    print(f"  Translational Mean: {mean_trans:.4f} m")
    print(f"  Rotational RMSE: {rmse_rot:.4f} deg")
    print(f"  Rotational Mean: {mean_rot:.4f} deg")

    # Extract metadata for CSV
    run_id = os.path.basename(args.estimated_file).replace(".txt", "")
    dataset = os.path.basename(args.groundtruth_file).replace(".txt", "").replace(".csv", "")

    # Save to CSV
    if args.csv:
        write_rpe_metrics_to_csv(args.csv, run_id, dataset, trans_errors, rot_errors)
        print(f"Results saved to: {args.csv}")

    # Generate plot
    if args.plot:
        plot_errors(trans_errors, rot_errors, args.plot)
        print(f"Plot saved to: {args.plot}")

if __name__ == "__main__":
    main()