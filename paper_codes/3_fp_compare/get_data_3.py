#!/usr/bin/env python3
"""
Extract the first frame from each ROS bag and save as PNG.

Usage:
    python3 get_data_3.py --bags <bag1.bag> [<bag2.bag> ...] --output <output_dir>

Each bag yields one file: <scene_name>.png
"""

import argparse
import os
import sys

import rosbag
import cv2
from cv_bridge import CvBridge


def main():
    parser = argparse.ArgumentParser(description="Extract first frame from ROS bags")
    parser.add_argument("--bags", nargs="+", required=True, help="Path(s) to .bag file(s)")
    parser.add_argument("--output", required=True, help="Output directory for PNGs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    bridge = CvBridge()

    for bag_path in args.bags:
        if not os.path.isfile(bag_path):
            print(f"[SKIP] {bag_path} not a file")
            continue

        scene_name = os.path.splitext(os.path.basename(bag_path))[0]
        out_path = os.path.join(args.output, f"{scene_name}.png")

        print(f"Processing {bag_path} ...", end=" ", flush=True)
        try:
            bag = rosbag.Bag(bag_path, "r")
            for topic, msg, t in bag.read_messages():
                if topic.endswith("/image_raw"):
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                    if len(cv_img.shape) == 3:
                        cv_img = cv_img[..., 0]
                    cv2.imwrite(out_path, cv_img)
                    print(f"saved to {out_path}")
                    break
            else:
                print("no /image_raw messages found!")
            bag.close()
        except Exception as e:
            print(f"ERROR: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
