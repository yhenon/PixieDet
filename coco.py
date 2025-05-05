import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json  # Import the json library
from bbox import BBoxBatch

class COCOKeypointDataset(data.Dataset):
    def __init__(self, json_path, img_dir, preproc=None):
        """
        Args:
            json_path (str): Path to the COCO-style JSON annotation file.
            img_dir (str): Path to the directory containing the images.
            preproc (callable, optional): Preprocessing function pipeline.
        """
        self.preproc = preproc
        self.img_dir = img_dir

        # --- COCO Keypoint Settings ---
        self.num_kpts = 17
        # COCO keypoint order:
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        self.kpt_flip_map = (
            (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
        )
        # Pass the flip map to the preprocessing function if it needs it
        if self.preproc is not None and hasattr(self.preproc, 'kpt_flip_map'):
             self.preproc.kpt_flip_map = self.kpt_flip_map
        self.vis_kpt_branch = True

        print(f"Loading annotations from: {json_path}")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        self.images = {}
        self.annotations = {}
        self.img_ids = []

        # Process images
        for img_info in coco_data['images']:
            self.images[img_info['id']] = img_info
            self.annotations[img_info['id']] = [] # Initialize annotations list for each image

        # Process annotations
        person_cat_id = -1
        # Find the category ID for 'person' (robust way)
        if 'categories' in coco_data:
            for cat in coco_data['categories']:
                if cat['name'] == 'person':
                    person_cat_id = cat['id']
                    print(f"Found 'person' category with ID: {person_cat_id}")
                    break
            if person_cat_id == -1:
                 print("Warning: 'person' category not found in JSON. Assuming category ID 1.")
                 person_cat_id = 1 # Default assumption if not found
        else:
            print("Warning: 'categories' key not found in JSON. Assuming category ID 1 for person.")
            person_cat_id = 1 # Default assumption if not found


        valid_annotations_count = 0
        images_with_persons = set()
        for ann in coco_data['annotations']:
            # Filter for 'person' category and annotations with keypoints
            #if ann['category_id'] == person_cat_id and 'keypoints' in ann and ann.get('num_keypoints', 0) > 0 and not ann.get('iscrowd', False):
            if ann['category_id'] == person_cat_id and 'keypoints' in ann and not ann.get('iscrowd', False):
                 # Add annotation to the corresponding image's list
                 img_id = ann['image_id']
                 if img_id in self.annotations:
                     self.annotations[img_id].append(ann)
                     images_with_persons.add(img_id)
                     valid_annotations_count += 1

        # Only keep image IDs that have valid person annotations
        self.img_ids = sorted(list(images_with_persons))

        print(f"Loaded {len(self.img_ids)} images with {valid_annotations_count} person instances.")
        if len(self.img_ids) == 0:
             print("Error: No valid images with person keypoint annotations found!")


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.images[img_id]
        img_anns = self.annotations[img_id]

        # Construct image path
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError

        height, width, _ = img.shape
        num_persons = len(img_anns)
        bboxes = np.zeros((num_persons, 4), dtype=np.float32)
        kpts = np.zeros((num_persons, self.num_kpts * 2), dtype=np.float32)
        kpts_vis = np.zeros((num_persons, self.num_kpts), dtype=np.float32)

        for idx, ann in enumerate(img_anns):
            # Bbox: COCO format [x, y, width, height] -> [x1, y1, x2, y2]
            bbox_coco = ann['bbox']
            x1 = bbox_coco[0]
            y1 = bbox_coco[1]
            x2 = x1 + bbox_coco[2]
            y2 = y1 + bbox_coco[3]
            bboxes[idx, :] = [x1, y1, x2, y2]

            # Keypoints: COCO format [x1, y1, v1, x2, y2, v2, ...]
            kpts_coco = ann['keypoints']
            for kpt_idx in range(self.num_kpts):
                x_coord = kpts_coco[kpt_idx * 3]
                y_coord = kpts_coco[kpt_idx * 3 + 1]
                visibility = kpts_coco[kpt_idx * 3 + 2] # 0: not labeled, 1: labeled but not visible, 2: labeled and visible
                kpts[idx, kpt_idx * 2] = x_coord
                kpts[idx, kpt_idx * 2 + 1] = y_coord
                kpts_vis[idx, kpt_idx] = 1.0 if visibility > 0 else 0.0

        if self.preproc is not None:
            img, bboxes, kpts, kpts_vis = self.preproc(img, bboxes, kpts, kpts_vis)
        kpts_vis = np.repeat(kpts_vis, 2, axis=1)
        boxes_xywh = bboxes
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        boxes_xywh = torch.from_numpy(boxes_xywh)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, BBoxBatch(boxes_xywh.unsqueeze(0), fmt='xywh'), kpts, kpts_vis
