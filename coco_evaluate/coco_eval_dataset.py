import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json

class COCOKeypointEvalDataset(data.Dataset):
    """
    COCO-style Keypoint Dataset Loader FOR EVALUATION.
    Loads images, applies *minimal* validation preprocessing (e.g., resize, normalize).
    Returns processed image, original image info, and image ID for the evaluator.
    """
    def __init__(self, json_path, img_dir, normalize, num_kpts=17): # normalize for eval (resize, normalize)
        """
        Args:
            json_path (str): Path to COCO-style JSON annotation file.
            img_dir (str): Path to image directory.
            normalize (callable): Validation normalizeessing pipeline.
                                Should accept (img) and return (processed_img_tensor).
                                It should ideally preserve aspect ratio for accurate eval.
            num_kpts (int): Number of keypoints (used for sanity checks).
        """
        self.normalize = normalize
        if self.normalize is None:
            raise ValueError("Preprocessing pipeline ('preproc') is required for eval dataset (for resize/normalize).")

        self.img_dir = img_dir
        self.num_kpts = num_kpts # Store but not directly used in __getitem__

        print(f"Loading EVALUATION annotations index from: {json_path}")
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        self.images = {} # Store image info only (filename, h, w)
        self.img_ids = [] # Store all image IDs present in the JSON

        if 'images' not in coco_data:
             raise ValueError(f"'images' key not found in {json_path}")

        # Process images - store all image IDs from the JSON 'images' list
        # The COCO API evaluation handles images with no GT annotations.
        for img_info in coco_data['images']:
            self.images[img_info['id']] = img_info
            self.img_ids.append(img_info['id'])

        self.img_ids = sorted(self.img_ids)

        print(f"Found {len(self.img_ids)} images listed in the evaluation JSON.")
        if not self.img_ids:
             print("Warning: No images found in the evaluation JSON 'images' list!")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        Returns:
            img_tensor (torch.Tensor): Preprocessed image tensor for model input.
            img_info (tuple): Original (height, width) of the image.
            img_id (int): The COCO image ID.
        """
        img_id = self.img_ids[index]
        img_info_dict = self.images[img_id]

        img_path = os.path.join(self.img_dir, img_info_dict['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Failed to load image {img_path}. Returning None/dummy data.")
            # Handle error: maybe return None and filter later, or raise
            # Returning dummy data might be problematic. Raising is safer.
            raise FileNotFoundError(f"Image not found or failed to load: {img_path}")


        height, width, _ = img.shape
        img_info = (height, width) # Store original height, width

        # Apply validation preprocessing (e.g., resize, normalize)
        # This preproc should NOT perform augmentations like flips etc.
        img_tensor = torch.from_numpy(self.normalize(img))
        return img_tensor, img_info, img_id