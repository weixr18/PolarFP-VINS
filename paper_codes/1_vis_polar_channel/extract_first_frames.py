#!/usr/bin/env python3
"""
Extract the first frame from ROS bags and save as PNG.

Usage:
    python3 extract_first_frames.py <bag_path> [<bag_path> ...] <output_dir>

Extracts the first /image_raw message from each bag and saves
as <output_dir>/<scene_name>.png (scene_name derived from the bag filename).
"""

import argparse
import os
import sys

try:
    import rosbag
    from cv_bridge import CvBridge
except ImportError:
    print("Error: ROS Python libraries not found.", file=sys.stderr)
    print("Source your ROS environment: source /opt/ros/noetic/setup.bash", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Extract first frame from ROS bags")
    parser.add_argument("bag_paths", nargs="+", help="Path(s) to .bag file(s)")
    parser.add_argument("output_dir", help="Directory to save PNG frames")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bridge = CvBridge()
    import cv2

    for bag_path in args.bag_paths:
        if not os.path.isfile(bag_path):
            print(f"Skipping: {bag_path} (not a file)")
            continue

        scene_name = os.path.splitext(os.path.basename(bag_path))[0]
        output_path = os.path.join(args.output_dir, f"{scene_name}.png")

        print(f"Processing {bag_path} ...", end=" ")
        try:
            bag = rosbag.Bag(bag_path, "r")
            for topic, msg, t in bag.read_messages():
                if topic.endswith("/image_raw"):
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                    cv_img = cv_img[..., 0] if len(cv_img.shape) == 3 else cv_img
                    cv2.imwrite(output_path, cv_img)
                    print(f"saved to {output_path}")
                    break
            else:
                print("no /image_raw topic found!")
            bag.close()
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
