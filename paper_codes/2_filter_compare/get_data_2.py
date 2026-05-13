#!/usr/bin/env python3
"""
Extract K random frames from each ROS bag and save as PNG.

Usage:
    python3 extract_random_frames.py --bags <bag1.bag> [<bag2.bag> ...] --output <output_dir> --k 10

Each bag yields files named: <scene_name>_<idx>.png  (idx = 0..K-1)
"""

import argparse
import os
import random
import sys

import rosbag
import cv2
from cv_bridge import CvBridge


def main():
    parser = argparse.ArgumentParser(description="Extract K random frames from ROS bags")
    parser.add_argument("--bags", nargs="+", required=True, help="Path(s) to .bag file(s)")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    parser.add_argument("--k", type=int, default=10, help="Number of random frames per bag (default: 10)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    bridge = CvBridge()

    for bag_path in args.bags:
        if not os.path.isfile(bag_path):
            print(f"[SKIP] {bag_path} not a file")
            continue

        scene_name = os.path.splitext(os.path.basename(bag_path))[0]

        print(f"Scanning {bag_path} ...", end=" ", flush=True)
        try:
            bag = rosbag.Bag(bag_path, "r")
        except Exception as e:
            print(f"ERROR opening bag: {e}")
            continue

        # Collect all image message timestamps
        timestamps = []
        try:
            for topic, msg, t in bag.read_messages():
                if topic.endswith("/image_raw"):
                    timestamps.append((t, msg))
        finally:
            bag.close()

        total = len(timestamps)
        if total == 0:
            print("no /image_raw messages found!")
            continue
        print(f"{total} frames")

        # Sample K random frames
        k = min(args.k, total)
        sampled = random.sample(timestamps, k)

        for idx, (t, msg) in enumerate(sorted(sampled, key=lambda x: x[0])):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            if len(cv_img.shape) == 3:
                cv_img = cv_img[..., 0]
            out_path = os.path.join(args.output, f"{scene_name}_{idx:02d}.png")
            cv2.imwrite(out_path, cv_img)
            print(f"  [{idx+1:02d}/{k}] {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
