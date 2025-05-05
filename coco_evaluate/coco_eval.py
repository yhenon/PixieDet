# Copyright (c) Megvii, Inc. and its affiliates.
# Adapted for custom keypoint model and dataset

import contextlib
import io
import itertools
import json
import tempfile
import time
import os
from collections import ChainMap, defaultdict
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
import cv2

import numpy as np
import torch

from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms 


# --- Make sure COCO API is installed ---
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("ERROR: pycocotools not found. Please run `pip install pycocotools`")
    COCO = None
    COCOeval = None
# -----------------------------------------

class CocoPoseEvaluator:
    """
    COCO OKS AP Evaluation class for Keypoints.
    Adapted for models with custom decoding and datasets loading JSON directly.
    """

    def __init__(
        self,
        json_file: str,      # Path to COCO-style ground truth annotation JSON
        cfg,
        confthre: float,
        nmsthre: float,
        num_keypoints: int, # Number of keypoints model predicts
        # --- Parameters for YOUR decoding ---
        # Add any parameters needed for your specific decoding functions
        # Example: variance, top_k, keep_top_k etc.
        decode_top_k=750,           # Example parameter
        decode_keep_top_k=500,      # Example parameter
        # ----------------------------------
        testdev: bool = False,
    ):
        """
        Args:
            json_file (str): Path to the COCO-style annotation file (e.g., instances_val2017.json).
            img_size (tuple): Image size (height, width) the network expects.
            confthre (float): Confidence threshold for filtering detections (used in custom decode).
            nmsthre (float): IoU threshold for NMS (used in custom decode).
            num_keypoints (int): Number of keypoints per instance.
            decode_variance (list): Variance for decoding (example parameter).
            decode_top_k (int): Keep top-k detections before NMS (example parameter).
            decode_keep_top_k (int): Keep top-k detections after NMS (example parameter).
            testdev (bool): Evaluate on test-dev dataset (if applicable).
        """
        # Ensure pycocotools are available
        if COCO is None or COCOeval is None:
            raise ImportError("pycocotools is not installed. Please run `pip install pycocotools`")

        self.cfg = cfg
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_keypoints = num_keypoints
        self.testdev = testdev
        self.num_classes = 1 # Hardcoded for person keypoint detection
        self.person_category_id = 1 # Standard COCO ID for 'person'

        # --- Store decoding parameters ---
        self.decode_variance = cfg.variance
        self.decode_top_k = decode_top_k
        self.decode_keep_top_k = decode_keep_top_k
        # --- Add other parameters as needed ---

        # --- Load Ground Truth Annotations ---
        logger.info(f"Loading COCO ground truth from: {json_file}")
        self.cocoGt = COCO(json_file)
        # Verify person category ID exists in GT
        person_cats = self.cocoGt.getCatIds(catNms=['person'])
        if not person_cats:
            logger.warning("'person' category not found in GT annotations. Using ID 1, but evaluation might fail.")
            self.person_category_id = 1
        else:
            self.person_category_id = person_cats[0]
            logger.info(f"Using 'person' category ID from GT: {self.person_category_id}")

    def evaluate(
        self, model, dataloader, return_outputs=False):
        """
        Run evaluation on the given dataloader.

        Args:
            model: The model to evaluate (should be in eval mode).
            dataloader: Dataloader providing (img_tensor, img_info, img_id).
            distributed (bool): Whether evaluation is distributed.
            half (bool): Whether to use half precision (FP16).
            return_outputs (bool): Whether to return raw formatted outputs.

        Returns:
            tuple: (oks_ap50_95, oks_ap50, summary_str) or (results, outputs_dict)
        """
        
        #model = model.eval()

        # Ensure priors are on the correct device (if used)
        priors_device = None
        self.priors = None
        if self.priors is not None:
             # Assume model is on cuda if available, else cpu
            device = next(model.parameters()).device
            self.priors = self.priors.to(device)
            priors_device = device
            logger.info(f"Priors moved to device: {priors_device}")


        data_list_coco_fmt = [] # Stores results for COCO API [{image_id, cat_id, bbox, score, keypoints}, ...]
        output_data_raw = defaultdict(list) # Stores raw decoded outputs per image_id if needed

        progress_bar = tqdm

        inference_time = 0
        decode_nms_time = 0
        n_samples = 0 # Count actual processed samples

        # --- Evaluation Loop ---
        for cur_iter, batch_data in enumerate(progress_bar(dataloader)):
            # Unpack batch data (handle potential variations in dataloader output)
            try:
                imgs, info_imgs_batch, ids_batch = batch_data
                if imgs.shape[0] == 1:
                    info_imgs_batch = [info_imgs_batch]
                # info_imgs_batch format: list of (h, w) tuples or similar
                # ids_batch format: list/tensor of image ids
            except ValueError:
                 logger.error(f"Error unpacking batch data at iter {cur_iter}. Expected (imgs, info_imgs, ids). Skipping batch.")
                 continue

            batch_size = imgs.shape[0]
            n_samples += batch_size # Accumulate processed samples count

            # --- Move data to device ---
            device = next(model.parameters()).device # Get model's device
            imgs = imgs.to(device)
            #print(imgs.shape, imgs.min(), imgs.max())

            with torch.no_grad():
                # --- Inference ---
                time_start = time.time()
                # Adapt this call based on your model's forward signature
                # Example: raw_outputs = model(imgs)
                # Example: loc, conf, landms = model(imgs)
                # Assume model returns tuple: (locations, confidences, landmarks)
                im_height = imgs.shape[2]
                im_width = imgs.shape[3]
                priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
                priors = priorbox.forward()
                priors = priors.to(device)
                raw_outputs = model(imgs)
                infer_end = time.time()
                inference_time += infer_end - time_start
                outputs_batch_decoded = self.decode_and_nms_batch(
                    raw_outputs, # loc, conf, landms (or combined)
                    priors_device, # Pass device for tensor creation if needed
                    info_imgs_batch, # Original image sizes for scaling
                    priors
                )

                decode_nms_end = time.time()
                decode_nms_time += decode_nms_end - infer_end
                # --- Format results for COCO API ---
                data_list_batch = self.convert_to_coco_keypoint_format(
                    outputs_batch_decoded, info_imgs_batch, ids_batch, imgs
                )
                data_list_coco_fmt.extend(data_list_batch)

                # Store raw outputs if requested
                if return_outputs:
                     for i, img_id in enumerate(ids_batch):
                         if outputs_batch_decoded[i] is not None:
                              output_data_raw[int(img_id)].append(outputs_batch_decoded[i].cpu().numpy()) # Example: store as numpy


        # --- Gather results in distributed setting ---
        total_samples = n_samples * dataloader.batch_size if n_samples > 0 else 0 # Approx total samples
        statistics = torch.cuda.FloatTensor([inference_time, decode_nms_time, total_samples]) # Use total processed samples

        # --- Perform COCO Evaluation (only on main process) ---
        eval_results = self.evaluate_prediction(data_list_coco_fmt, statistics)

        if return_outputs:
             # Ensure output_data_raw is returned only by main process if distributed
                return eval_results, output_data_raw
        return eval_results # Return tuple: (ap50_95, ap50, summary_str)


    def decode_and_nms_batch(self, raw_outputs, device, info_imgs_batch, priors):
        """
        Placeholder for YOUR custom decoding and NMS logic.

        Args:
            raw_outputs: The direct output from model(imgs) (e.g., tuple of loc, conf, landms).
            device: The device where tensors should be processed.
            info_imgs_batch: List of original (height, width) for each image in the batch.

        Returns:
            list: A list (length batch_size) where each element is either:
                  - A Tensor: [N_dets, 7 + 3*Nk] with format
                    [x1,y1,x2,y2, obj_sc, cls_sc, cls_idx, kpt1_v,..kN_v, kpt1_x,..kN_x, kpt1_y,..kN_y]
                  - None: if no detections found for that image.
                  OR adapt to a simpler format if sufficient, e.g.,
                  - A Tensor: [N_dets, 5 + 3*Nk] with format
                    [x1, y1, x2, y2, score, kpt1_x, kpt1_y, kpt1_vis, ..., kptN_x, kptN_y, kptN_vis]
        """
        batch_size = raw_outputs[0].shape[0] # Assuming loc is first element
        outputs_batch_decoded = [None] * batch_size

        loc_batch, conf_batch, landms_batch, landms_vis_batch = raw_outputs # Assuming this structure

        # --- Ensure priors are on the correct device ---
        priors_data = priors.to(loc_batch.device) # Ensure priors match batch device

        for i in range(batch_size):
            # --- Get data for the i-th image ---
            loc = loc_batch[i]         # Shape: [num_priors, 4]
            conf = conf_batch[i]       # Shape: [num_priors, num_classes] (e.g., num_classes=2 for bg/fg)
            landms = landms_batch[i]   # Shape: [num_priors, num_keypoints * 2]
            landms_vis = landms_vis_batch[i]
            im_height, im_width = info_imgs_batch[i] # Original H, W for this image

            # --- 1. Decode Boxes ---
            # Create scale tensor for this image's original dimensions
            scale = torch.tensor([im_width, im_height, im_width, im_height], device=loc.device)
            # Decode boxes relative to priors
            boxes = decode(loc, priors_data, self.decode_variance) # Use variance from init
            # Scale boxes to original image dimensions
            boxes = boxes * scale # Scale to original image size
            # Let's assume decode outputs coords relative to INPUT size (e.g., self.img_size)
            # We will scale back to original size in convert_to_coco_keypoint_format

            # --- 2. Decode Landmarks (Keypoints) ---
            # Create scale tensor for landmarks (x, y repeated)
            scale_landm = torch.tensor([im_width, im_height] * self.num_keypoints, device=loc.device)
            landms_decoded = decode_landm(landms, priors_data, self.decode_variance)
            # Scale landmarks to original image dimensions (similar caveat as boxes)
            landms_decoded = landms_decoded * scale_landm # Scale to original image size
            # Assume decode_landm outputs coords relative to INPUT size

            # --- 3. Get Scores and Apply Confidence Threshold ---
            # Assuming conf shape is [num_priors, 2] (bg, fg) and we want the foreground score
            scores = conf[:, 1]
            score_mask = scores > self.confthre
            boxes = boxes[score_mask]
            scores = scores[score_mask]
            landms_decoded = landms_decoded[score_mask]
            landms_vis = landms_vis[score_mask]

            if boxes.shape[0] == 0:
                continue # No detections above threshold for this image

            # --- 4. NMS ---
            # Combine boxes and scores for NMS
            dets = torch.cat((boxes, scores.unsqueeze(1)), dim=1)
            # Ensure dets is on CPU and numpy for py_cpu_nms
            dets_np = dets.cpu().numpy()
            landms_np = landms_decoded.cpu().numpy()
            landms_vis_np = landms_vis.cpu().numpy()

            keep_indices = py_cpu_nms(dets_np, self.nmsthre)
            dets_nms = dets_np[keep_indices]
            landms_nms = landms_np[keep_indices]
            landms_vis_nms = landms_vis_np[keep_indices]

            # --- 5. Keep Top-K ---
            # (Optional, based on your original decoding snippet)
            # You had top_k *before* NMS and keep_top_k *after* NMS.
            # Applying keep_top_k after NMS:
            if len(keep_indices) > self.decode_keep_top_k:
                dets_nms = dets_nms[:self.decode_keep_top_k]
                landms_nms = landms_nms[:self.decode_keep_top_k]
                landms_vis_nms = landms_vis_nms[:self.decode_keep_top_k]


            # --- 6. Format Output ---
            # We need to construct the [N_dets, 7 + 3*Nk] tensor
            # [x1,y1,x2,y2, obj_sc, cls_sc, cls_idx, kpt1_v,..kN_v, kpt1_x,..kN_x, kpt1_y,..kN_y]

            num_dets_final = dets_nms.shape[0]
            if num_dets_final == 0:
                continue

            # Convert back to tensor if needed
            final_boxes = torch.from_numpy(dets_nms[:, :4]).to(device)
            final_scores = torch.from_numpy(dets_nms[:, 4]).to(device) # This is the detection score
            final_landms = torch.from_numpy(landms_nms).to(device) # Shape [N_dets, Nk*2]
            final_landms_vis = torch.from_numpy(landms_vis_nms).to(device) # Shape [N_dets, Nk*2]

            # Prepare the output tensor
            # Size: num_dets x (4 + 1 + 1 + 1 + Nk + Nk*2) = num_dets x (7 + 3*Nk)
            output_tensor = torch.zeros((num_dets_final, 7 + 3 * self.num_keypoints), device=device)

            output_tensor[:, 0:4] = final_boxes        # x1, y1, x2, y2
            output_tensor[:, 4] = final_scores         # Use detection score as objectness score
            output_tensor[:, 5] = torch.ones(num_dets_final, device=device) # Class score (assume 1.0 for person)
            output_tensor[:, 6] = torch.zeros(num_dets_final, device=device) # Class index (0 internally)

            output_tensor[:, 7 : 7 + self.num_keypoints] = final_landms_vis

            # Reshape landmarks [N_dets, Nk*2] -> [N_dets, Nk, 2] -> extract x, y
            final_landms_reshaped = final_landms.view(num_dets_final, self.num_keypoints, 2)
            kpt_x_coords = final_landms_reshaped[:, :, 0] # [N_dets, Nk]
            kpt_y_coords = final_landms_reshaped[:, :, 1] # [N_dets, Nk]

            output_tensor[:, 7 + self.num_keypoints : 7 + 2 * self.num_keypoints] = kpt_x_coords
            output_tensor[:, 7 + 2 * self.num_keypoints : ] = kpt_y_coords

            outputs_batch_decoded[i] = output_tensor
                
        return outputs_batch_decoded


    def convert_to_coco_keypoint_format(self, outputs_batch, info_imgs_batch, ids_batch, imgs=None):
        """
        Converts decoded detections (including keypoints) to COCO keypoint result format.
        Assumes outputs_batch contains tensors in the format:
        [N_dets, x1,y1,x2,y2, obj_sc, cls_sc, cls_idx, kpt1_v,..kN_v, kpt1_x,..kN_x, kpt1_y,..kN_y]
        Coordinates are assumed to be relative to the network input size (self.img_size).
        """
        data_list = []

        for i, output_per_image in enumerate(outputs_batch):
            if output_per_image is None or output_per_image.shape[0] == 0:
                continue
            img_h, img_w = info_imgs_batch[i] # Original height/width for this image
            img_id = ids_batch[i]

            output_per_image = output_per_image.cpu() # Move to CPU for processing

            # Extract data columns based on the assumed format
            bboxes_xyxy = output_per_image[:, 0:4]
            # Combine objectness and class score (or just use objectness score)
            obj_scores = output_per_image[:, 4]
            cls_scores = output_per_image[:, 5]
            scores = obj_scores * cls_scores # Or just obj_scores if cls_score is dummy

            # cls_ids_internal = output_per_image[:, 6] # Internal index (should be 0)
            kpt_vis_scores = output_per_image[:, 7 : 7 + self.num_keypoints]
            kpt_x_coords = output_per_image[:, 7 + self.num_keypoints : 7 + 2 * self.num_keypoints]
            kpt_y_coords = output_per_image[:, 7 + 2 * self.num_keypoints : ]

            # Reshape keypoints for easier access: [N_dets, Nk, 2]
            kpt_coords = torch.stack((kpt_x_coords, kpt_y_coords), dim=-1)

            # --- Scale boxes and keypoints back to ORIGINAL image dimensions ---
            # The decoded outputs are relative to network input size (net_h, net_w)
            # Remove padding (if letterboxing was used) and scale back
            # Calculate padding (assuming centered padding)
            scale = 1
            pad_x = 0
            pad_y = 0
            #pad_x = (net_w - img_w * scale) / 2
            #pad_y = (net_h - img_h * scale) / 2

            bboxes_xyxy[:, [0, 2]] -= pad_x # Adjust x coords for padding
            bboxes_xyxy[:, [1, 3]] -= pad_y # Adjust y coords for padding
            bboxes_xyxy /= scale           # Scale back to original size

            # Back to image scale
            #bboxes_xyxy[:, [0, 2]] *= img_w.item()
            #bboxes_xyxy[:, [1, 3]] *= img_h.item()
            #kpt_coords[:, ::3] *= img_w.item()
            #kpt_coords[:, 1::3] *= img_h.item()

            # Clip boxes to image boundaries
            bboxes_xyxy[:, [0, 2]] = bboxes_xyxy[:, [0, 2]].clamp(0, img_w.item())
            bboxes_xyxy[:, [1, 3]] = bboxes_xyxy[:, [1, 3]].clamp(0, img_h.item())

            kpt_coords[:, :, 0] -= pad_x # Adjust x coords for padding
            kpt_coords[:, :, 1] -= pad_y # Adjust y coords for padding
            kpt_coords /= scale          # Scale back to original size

            # Convert boxes to XYWH format for COCO
            bboxes_xywh = xyxy2wh(bboxes_xyxy) # Use local xyxy2wh if needed

            num_dets = output_per_image.shape[0]
            imgv = imgs[0,:,:,:].cpu()
            imgv *= torch.from_numpy(np.array([0.2290, 0.2240, 0.2250])).unsqueeze(1).unsqueeze(2)
            imgv += torch.from_numpy(np.array([0.4850, 0.4560, 0.4060])).unsqueeze(1).unsqueeze(2)
            imgv = imgv * 255.0
            imgv = imgv.permute(1,2,0)
            imgv = imgv.cpu().numpy().astype(np.uint8).copy()
            
            for det_idx in range(num_dets):
                # Class ID for COCO is always 'person'
                class_id = self.person_category_id

                keypoints_coco_fmt = []
                kpts_det = kpt_coords[det_idx] # [Nk, 2]
                kpt_scores_det = kpt_vis_scores[det_idx] # [Nk]

                for kp_idx in range(self.num_keypoints):
                    x, y = kpts_det[kp_idx].tolist()
                    score_kpt = kpt_scores_det[kp_idx].item()

                    # Convert keypoint score/visibility to COCO format (0, 1, 2)
                    # Visibility: 0=not labeled, 1=labeled but occluded, 2=labeled and visible
                    # Use the score directly if it represents visibility/confidence well enough,
                    # OR apply a threshold. A simple threshold is common:
                    visibility = 0
                    if score_kpt > 0.1: # Example threshold (tune this!)
                        visibility = 2 # Treat as visible if score is high enough
                    # If your kpt_vis_scores *already* represent 0,1,2, use them directly:
                    # visibility = int(round(score_kpt)) # If scores are already 0, 1, or 2

                    # Format: x, y, v (rounded for smaller JSON)
                    keypoints_coco_fmt.extend([round(x, 2), round(y, 2), visibility])
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": class_id,
                    "bbox": [round(c, 2) for c in bboxes_xywh[det_idx].numpy().tolist()],
                    "score": round(scores[det_idx].numpy().item(), 5), # Use combined score
                    "keypoints": keypoints_coco_fmt,
                }
                bbox = [int(x) for x in pred_data['bbox']]
                if pred_data['score'] < 0.8:
                    continue
                kpts = [int(x) for x in pred_data["keypoints"]]
                #for i in range(17):
                #    cv2.circle(imgv, (kpts[i*3], kpts[i*3+1]), 1, (0,0,255))
                #cv2.rectangle(imgv, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)
                data_list.append(pred_data)
            #cv2.imshow('x', imgv)
            #cv2.waitKey(0)
        return data_list


    def evaluate_prediction(self, data_dict, statistics):
        """
        Evaluates the predictions using COCOeval for keypoints.
        (Largely unchanged, uses self.cocoGt loaded in __init__)
        """
        logger.info("Evaluating keypoints in main process...")

        iouType = "keypoints" # Explicitly use keypoints for OKS

        # Handle case where statistics might be None if no samples processed
        if statistics is None or len(statistics) < 3:
            inference_time = 0
            nms_time = 0
            n_samples = 0
            info = "No samples processed or statistics unavailable.\n"
        else:
            inference_time = statistics[0].item()
            nms_time = statistics[1].item() # This is now decode + NMS time
            n_samples = statistics[2].item() # Use actual processed sample count

            # Guard against division by zero if n_samples is 0
            if n_samples > 0:
                 # Note: dataloader batch size isn't directly available here,
                 # use n_samples which is the total count from statistics
                 a_infer_time = 1000 * inference_time / n_samples
                 a_decode_nms_time = 1000 * nms_time / n_samples
                 time_info = ", ".join(
                     [
                         "Average {} time: {:.2f} ms".format(k, v)
                         for k, v in zip(
                             ["forward", "decode+NMS", "total"],
                             [a_infer_time, a_decode_nms_time, (a_infer_time + a_decode_nms_time)],
                         )
                     ]
                 )
                 info = time_info + "\n"
            else:
                 info = "No samples processed.\n"

        if len(data_dict) == 0:
            logger.warning("No predictions generated, returning zero AP.")
            # Ensure cocoGt is available before attempting evaluation
            if self.cocoGt is None:
                 return 0, 0, info + "\nGround truth not loaded."
            # Still run summary to get the header, but results will be 0
            cocoEval = COCOeval(self.cocoGt, self.cocoGt.loadRes([]), iouType) # Empty results
            cocoEval.params.catIds = [self.person_category_id]
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                 cocoEval.summarize()
            summary = redirect_string.getvalue()
            logger.info(summary) # Log the summary even with zeros
            return 0, 0, info + "\n" + summary


        # Load results (predictions) into COCO API
        # Use temp file to load results
        _, tmp = tempfile.mkstemp()
        try:
            logger.info(f"Saving {len(data_dict)} predictions to temporary file for COCO eval...")
            with open(tmp, "w") as f:
                json.dump(data_dict, f)
            # Check file size
            file_size = os.path.getsize(tmp)
            logger.info(f"Temporary prediction file size: {file_size / 1024:.2f} KB")
            if file_size == 0 and len(data_dict) > 0:
                logger.warning("Prediction file is empty despite having data_dict entries!")

            cocoDt = self.cocoGt.loadRes(tmp)
            logger.info("Predictions loaded into COCO API.")

        except Exception as e:
            logger.error(f"Error loading prediction results into COCO API: {e}")
            if os.path.exists(tmp): os.remove(tmp)
            return 0, 0, info + f"\nError loading results: {e}"
        finally:
            if os.path.exists(tmp): os.remove(tmp)


        # --- Use COCOeval ---
        cocoEval = COCOeval(self.cocoGt, cocoDt, iouType) # Use iouType = "keypoints"

        # Evaluate only for the 'person' category
        cocoEval.params.catIds = [self.person_category_id]
        # You can adjust other params like area ranges if needed
        # cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]] # all, medium, large

        logger.info("Running COCO evaluation...")
        cocoEval.evaluate()
        logger.info("Accumulating results...")
        cocoEval.accumulate()

        redirect_string = io.StringIO()
        logger.info("Summarizing results...")
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize() # Prints the standard COCO keypoint AP summary

        summary = redirect_string.getvalue()
        info += summary # Append summary to the timing info
        logger.info(f"\n{summary}") # Log the summary table

        # Extract key metrics (OKS AP @ IoU=0.50:0.95 and OKS AP @ IoU=0.50)
        try:
             ap50_95 = cocoEval.stats[0] # AP @ OKS=0.50:0.95
             ap50 = cocoEval.stats[1]    # AP @ OKS=0.50
        except IndexError:
             logger.error("Could not extract AP scores from cocoEval.stats.")
             ap50_95 = 0.0
             ap50 = 0.0

        return ap50_95, ap50, info # Return AP50-95, AP50, and full summary string

# --- Helper function needed by convert_to_coco_keypoint_format ---
def xyxy2wh(bboxes_xyxy):
    """Converts nx4 boxes from [x1, y1, x2, y2] to [x1, y1, w, h]"""
    if isinstance(bboxes_xyxy, torch.Tensor):
        bboxes_xywh = torch.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0]  # x1
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1]  # y1
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]  # width
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]  # height
    else: # Assume numpy
        bboxes_xywh = np.zeros_like(bboxes_xyxy)
        bboxes_xywh[:, 0] = bboxes_xyxy[:, 0]  # x1
        bboxes_xywh[:, 1] = bboxes_xyxy[:, 1]  # y1
        bboxes_xywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]  # width
        bboxes_xywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]  # height
    return bboxes_xywh