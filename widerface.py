import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from bbox import BBoxBatch
import cv2


class WiderFaceDataset(Dataset):
    def __init__(self, root, label_file, transforms=None):
        """
        Args:
            root (str): path to the WIDER train/ folder
            label_file (str): path to train/label.txt
            transforms (callable, optional): transforms(img, target) -> img, target
        """
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.transforms = transforms

        # read and parse the label file
        self.samples = []  # list of (image_path, ann_list)
        with open(label_file, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        i = 0
        while i < len(lines):
            # detect image line (may start with '#' or not)
            line = lines[i]
            if ".jpg" not in line:
                i += 1
                continue
            img_rel = line.lstrip('# ').strip().split(' ')[0]
            img_path = os.path.join(self.img_dir, img_rel)
            i += 1

            ann_list = []
            # consume following lines until next image
            while i < len(lines) and ".jpg" not in lines[i]:
                parts = lines[i].split()
                # expected 4 bbox + (5 keypoints × 3 values)  = 19 numbers
                if len(parts) == 19:
                    nums = list(map(float, parts))
                    # bbox: x1, y1, x2, y2
                    bbox = nums[0:4]
                    # keypoints: 5 points, each (x, y, v)
                    kps = []
                    for k in range(5):
                        x = nums[4 + k*3 + 0]
                        y = nums[4 + k*3 + 1]
                        v = int(nums[4 + k*3 + 2])  # visibility flag
                        v = 1 if v >= 0  else 0
                        kps.append((x, y, v))
                    score = nums[-1]
                    ann_list.append({
                        "bbox": bbox,
                        "keypoints": kps,
                        "score": score
                    })
                else:
                    raise ValueError
                i += 1
            self.samples.append((img_path, ann_list))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann_list = self.samples[idx]
        # load image
        img = Image.open(img_path).convert("RGB")

        # build boxes tensor (x, y, width, height)
        boxes_x1y1x2y2 = np.array([ann["bbox"] for ann in ann_list])

        # format: (x, y, v)
        keypoints = np.zeros((len(boxes_x1y1x2y2), 5*2))
        keypoints_vis = np.zeros((len(boxes_x1y1x2y2), 5))
        for o, ann in enumerate(ann_list):
            for k, (x, y, v) in enumerate(ann["keypoints"]):
                keypoints[o, 2*k] = x
                keypoints[o, 2*k+1] = y
                keypoints_vis[o, k] = v

        img_np = np.asarray(img)
        if self.transforms is not None and boxes_x1y1x2y2.shape[0] > 0:
            img_np, boxes_x1y1x2y2, keypoints, keypoints_vis = self.transforms(img_np, boxes_x1y1x2y2, keypoints, keypoints_vis)
        else:
            boxes_x1y1x2y2 = np.zeros((0, 4))
            img_np = cv2.resize(img_np, (640, 640))

        keypoints_vis = np.repeat(keypoints_vis, 2, axis=1)
        boxes_xywh = boxes_x1y1x2y2
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        boxes_xywh = torch.from_numpy(boxes_xywh)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_np, BBoxBatch(boxes_xywh.unsqueeze(0), fmt='xywh'), keypoints, keypoints_vis
