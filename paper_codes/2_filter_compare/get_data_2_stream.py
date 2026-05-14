#!/usr/bin/env python3
"""
Extract first N consecutive frames from ROS bags and save as PNG.

Each bag yields files in: <output_dir>/<scene_name>/frame_00000.png ... frame_N-1.png

Usage:
    python3 get_data_2_stream.py <bag1.bag> [<bag2.bag> ...] <output_dir> [--n 200]
"""

import argparse
import os
import sys

import rosbag
import cv2
from cv_bridge import CvBridge


def main():
    parser = argparse.ArgumentParser(
        description="Extract first N consecutive frames from ROS bags")
    parser.add_argument("bags", nargs="+", help="Path(s) to .bag file(s)")
    parser.add_argument("output_dir", help="Output directory for PNGs")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of consecutive frames per bag (default: 200)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bridge = CvBridge()

    for bag_path in args.bags:
        if not os.path.isfile(bag_path):
            print(f"[SKIP] {bag_path} not a file")
            continue

        # Scene name from parent directory (e.g. "13-27-22")
        scene_name = os.path.basename(os.path.dirname(bag_path))

        print(f"Scanning {bag_path} (scene={scene_name}) ...", end=" ", flush=True)
        try:
            bag = rosbag.Bag(bag_path, "r")
        except Exception as e:
            print(f"ERROR opening bag: {e}")
            continue

        # Collect first N image messages in order
        frames = []
        try:
            for topic, msg, t in bag.read_messages():
                if topic == "/arena_cam_qc2/image_raw":
                    frames.append(msg)
                    if len(frames) >= args.n:
                        break
        finally:
            bag.close()

        total = len(frames)
        if total == 0:
            print("no /image_raw messages found!")
            continue
        print(f"{total} frames")

        # Output directory for this scene
        scene_dir = os.path.join(args.output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        # Save each frame
        for idx, msg in enumerate(frames):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            if len(cv_img.shape) == 3:
                cv_img = cv_img[..., 0]
            out_path = os.path.join(scene_dir, f"frame_{idx:05d}.png")
            cv2.imwrite(out_path, cv_img)

        print(f"  Saved {total} frames to {scene_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
