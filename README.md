# Exploring Image Masking for Adaptive Intelligent S.L.A.M. Results

## Computer Systems Intelligence Lab at SDSU

This repository contains the results and scripts used for analysis in the study Exploring Image Masking for Adaptive Intelligent S.L.A.M.<br>
Conducted by: Jia Gapuz<br>
Mentors: Dr. Bryan Donyanavard, Alles Rebel<br>
Supporting Program: STEM Pathways<br>

## Analysis

### Absolute Trajectory Error (ATE)
measures how far off the **estimated positions** (e.g., of a camera or robot) are from the **true positions** over time.

to produce the ATE results file, in the root folder run the command 

```python
python run_ate.py
```

### Relative Pose Error
measures the difference between an **estimated pose** and the **true pose** of an object (usually a camera or robot) at a specific point in time.

to produce the ATE results file, in the root folder run the command 

```python
python run_rpe.py
```

### Tracking Time
**how long it takes the system to estimate the current pose** (position + orientation) of the camera or robot **for a single frame**.
to produce the ATE results file, in the root folder run the command 

```python
python run_tracking_time.py
```

## Metrics

### Collected from running the simple static masks on ORB-SLAM3.
Located in the folder
```python
MaskingResults/
```
Link to repository: https://github.com/gapuzjia/ORB_SLAM3

## Datasets Used

### European Robotics Challenge (EuRoC)  - https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

visual-inertial datasets collected on-board a Micro Aerial Vehicle (MAV). The datasets contain stereo images, synchronized IMU measurements, and accurate motion and structure ground-truth. Located in folder: 

```jsx
EuRoCGroundTruth/
```
