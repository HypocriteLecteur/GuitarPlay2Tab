import json
import gzip
import numpy as np
import cv2

# Dealing with Python relative import issues
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from video_processing.utils.utils_math import line_line_intersection_batch

def format_and_store_json(detected_frames, total_frames, frame_width, frame_height, save_dir):
    keys = list(detected_frames.keys())
    data = {
        "info": {
            "number_of_frames": total_frames,
            "width": frame_width,
            "height": frame_height,
            "number_of_frets": detected_frames[keys[0]][0].frets.shape[0],
            "number_of_strings": detected_frames[keys[0]][0].strings.shape[0]
        },
        "annotations_fretboard": []
    }
    for key in keys:
        # make sure strings[0] is at the top
        if detected_frames[key][0].strings[0][0] < detected_frames[key][0].strings[1][0]:
            strings = detected_frames[key][0].strings
        else:
            strings = detected_frames[key][0].strings[[1, 0]]
        
        # make sure frets[0] is at the right-most
        if detected_frames[key][0].frets[0][0] < detected_frames[key][0].frets[-1][0]:
            frets = detected_frames[key][0].frets[::-1]
        else:
            frets = detected_frames[key][0].frets
        
        keypoints_top = line_line_intersection_batch(frets, strings[0])
        keypoints_bottom = line_line_intersection_batch(frets, strings[1])

        keypoints = np.zeros((2*frets.shape[0], 2))
        keypoints[0::2, :] = keypoints_top
        keypoints[1::2, :] = keypoints_bottom

        annotation = {
            "frame": key,
            "keypoints": keypoints.astype(int).tolist(),
            "oriented_bb": detected_frames[key][0].oriented_bb
        }
        data["annotations_fretboard"].append(annotation)
    
    output_path = str(save_dir / "annotations.json.gz")
    with gzip.open(output_path, 'wt', encoding='UTF-8') as zipfile:
        json.dump(data, zipfile)
    
    print(f"COCO JSON file with annotations saved to {output_path}")

def resize_with_aspect(frame, max_size):
    h, w = frame.shape[:2]
    scale = min(max_size[1] / w, max_size[0] / h)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA), scale
