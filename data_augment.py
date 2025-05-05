import cv2
import numpy as np
import random

def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def _crop(image, boxes, kpts_vis, keypoints, img_dim):
    """
    Crop the image and adjust boxes and keypoints. Handles scales > 1.0 by padding.

    Args:
        image: Input image [H, W, C]
        boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
        kpts_vis: Keypoint visibility [N, K] with values 0=not labeled, 1=labeled but not visible, 2=labeled and visible
        keypoints: Keypoint coordinates [N, K*2] in format [x1, y1, x2, y2, ..., xk, yk]
        img_dim: Target image dimension (used for filtering)

    Returns:
        Cropped/padded image, adjusted boxes, visibility, and keypoints, pad_flag
    """
    height, width, _ = image.shape
    pad_image_flag = True

    for _ in range(250):
        # Choose scale from predefined scales (can include > 1.0)
        scale = random.uniform(0.5, 1.5 )
        short_side = min(width, height)
        w = int(scale * short_side)
        h = w

        # Determine crop coordinates, handle scale > 1
        if w >= width:
            # If crop width is larger than image width, center it horizontally
            l = -(w - width) // 2
        else:
            # Original behavior: random placement
            l = random.randrange(width - w)

        if h >= height:
            # If crop height is larger than image height, center it vertically
            t = -(h - height) // 2
        else:
            # Original behavior: random placement
            t = random.randrange(height - h)

        roi = np.array((l, t, l + w, t + h)) # roi can now have negative coords

        # Check if any box has IoF >= 1 with the ROI (using original image coords)
        # Assuming matrix_iof handles the roi conceptually correctly
        value = matrix_iof(boxes, roi[np.newaxis])
        flag = (value >= 1)
        if not flag.any():
            continue

        # Filter boxes based on center point (using original image coords)
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        boxes_t = boxes[mask_a].copy()
        kpts_vis_t = kpts_vis[mask_a].copy()
        kpts_t = keypoints[mask_a].copy()

        if boxes_t.shape[0] == 0:
            continue

        # --- Crop/Pad the image ---
        # Create target image canvas (pad with zeros or a mean value)
        # Using zeros for simplicity. Change np.zeros to np.full with a mean value if needed.
        image_t = np.zeros((h, w, image.shape[2]), dtype=image.dtype)

        # Calculate source coordinates (valid region within the original image)
        src_x1 = max(0, l)
        src_y1 = max(0, t)
        src_x2 = min(width, l + w)
        src_y2 = min(height, t + h)

        # Calculate destination coordinates (where to place the source patch in image_t)
        dst_x1 = max(0, -l)
        dst_y1 = max(0, -t)
        # The size of the patch to copy determines the end coordinates implicitly
        # dst_x2 = dst_x1 + (src_x2 - src_x1)
        # dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Calculate the dimensions of the patch to copy
        copy_w = src_x2 - src_x1
        copy_h = src_y2 - src_y1

        # Copy the relevant part from the original image to the new canvas
        if copy_w > 0 and copy_h > 0:
            image_t[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
                image[src_y1:src_y2, src_x1:src_x2]

        # --- Adjust box coordinates ---
        # Clip boxes to the roi boundaries (relative to original image 0,0)
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        # Shift origin to the roi's top-left (l, t)
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] -= roi[:2]
        # Clip coordinates to be within the new image dimensions [0, w) and [0, h)
        # This is important if parts of the box fell into the padded area
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], 0)
        boxes_t[:, 2] = np.minimum(boxes_t[:, 2], w)
        boxes_t[:, 3] = np.minimum(boxes_t[:, 3], h)


        # --- Adjust keypoint coordinates ---
        # Reshape keypoints to [N, K, 2] for easier processing
        num_kpts = kpts_t.shape[1] // 2
        kpts_t_reshaped = kpts_t.reshape([-1, num_kpts, 2]).copy() # Ensure copy

        # Shift keypoint origin to the roi's top-left (l, t)
        kpts_t_reshaped[:, :, :] -= np.array(roi[:2])

        # Update visibility for keypoints that fall outside the *new* crop dimensions (w, h)
        for i in range(kpts_t_reshaped.shape[0]):  # For each instance
            for j in range(kpts_t_reshaped.shape[1]):  # For each keypoint
                # Skip keypoints that are not labeled (vis=0)
                if kpts_vis_t[i, j] == 0:
                    continue

                kpt_x, kpt_y = kpts_t_reshaped[i, j]

                # Check if keypoint is outside the NEW image boundaries (0..w-1, 0..h-1)
                # NOTE: Your original code set vis to 0. Docstring says 0=not labeled, 1=labeled but not visible.
                # If the point was visible (2) and goes out, maybe it should be 1?
                # Sticking to original code's behavior (setting to 0). Adjust if needed.
                if kpt_x < 0 or kpt_x >= w or kpt_y < 0 or kpt_y >= h:
                    # Set visibility based on your convention (using 0 as per original code)
                    kpts_vis_t[i, j] = 0
                    # Clip coordinates to be within the new image boundaries [0, w-1] or [0, h-1]
                    # Clip to w-1 and h-1 as coords are 0-based indices
                    kpts_t_reshaped[i, j, 0] = np.clip(kpts_t_reshaped[i, j, 0], 0, w - 1)
                    kpts_t_reshaped[i, j, 1] = np.clip(kpts_t_reshaped[i, j, 1], 0, h - 1)
                # else: # Point is inside the crop
                    # If point was originally labeled but not visible (1), keep it as 1?
                    # Or if it was visible (2), keep it as 2?
                    # Current logic doesn't change visibility if inside, which seems correct.
                    # Pass

        # Reshape keypoints back to original format [N, K*2]
        kpts_t = kpts_t_reshaped.reshape([-1, num_kpts * 2])

        # --- Ensure the box is large enough in the target dimension ---
        # Use the dimensions of the *output* image (w, h) for scaling check
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0]) # Width in pixels in cropped image
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1]) # Height in pixels in cropped image

        # Scale to target dimension - Check if this logic is still desired
        # Original logic used '+1' which might be slightly off for 0-based coords?
        # Let's recalculate based on w,h of the *new* image_t
        b_w_t_scaled = b_w_t / w * img_dim
        b_h_t_scaled = b_h_t / h * img_dim

        min_size_threshold = 0.0  # Original value was 0.0
        mask_b = np.minimum(b_w_t_scaled, b_h_t_scaled) > min_size_threshold

        boxes_t = boxes_t[mask_b]
        kpts_vis_t = kpts_vis_t[mask_b]
        kpts_t = kpts_t[mask_b]

        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, kpts_vis_t, kpts_t, pad_image_flag
    # If no suitable crop found after 250 attempts
    return image, boxes, kpts_vis, keypoints, pad_image_flag

def _distort(image):
    """Apply color distortion to image"""
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:
        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _mirror(image, boxes, keypoints, kpts_vis, kpt_flip_map):
    """
    Mirror image along with boxes and keypoints.
    
    Args:
        image: Input image
        boxes: Bounding boxes [N, 4]
        keypoints: Keypoint coordinates [N, K*2]
        kpts_vis: Keypoint visibility [N, K]
        kpt_flip_map: List of tuples (idx1, idx2) for swapping keypoints during flipping
    
    Returns:
        Flipped image, boxes, and keypoints
    """
    height, width, _ = image.shape

    if random.randrange(2):
        # Flip image horizontally
        image = image[:, ::-1].copy() 

        # Flip boxes horizontally
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        
        # Flip keypoints horizontally
        keypoints = keypoints.copy()
        num_kpts = keypoints.shape[1] // 2

        # Reshape to [N, K, 2] for easier processing
        kpts_reshaped = keypoints.reshape([-1, num_kpts, 2])
        
        # Flip x-coordinates of keypoints
        kpts_reshaped[:, :, 0] = width - kpts_reshaped[:, :, 0]

        # Swap left-right keypoints according to flip map
        if kpt_flip_map is not None:
            for idx1, idx2 in kpt_flip_map:
                # Swap keypoint coordinates
                tmp_kpt = kpts_reshaped[:, idx1, :].copy()
                kpts_reshaped[:, idx1, :] = kpts_reshaped[:, idx2, :]
                kpts_reshaped[:, idx2, :] = tmp_kpt
                
                # Swap visibility flags (if they exist)
                if kpts_vis is not None and kpts_vis.shape[1] > 0:
                    tmp_vis = kpts_vis[:, idx1].copy()
                    kpts_vis[:, idx1] = kpts_vis[:, idx2]
                    kpts_vis[:, idx2] = tmp_vis

        # Reshape back to original format [N, K*2]
        keypoints = kpts_reshaped.reshape([-1, num_kpts * 2])

    return image, boxes, keypoints, kpts_vis


def _pad_to_square(image, pad_image_flag):
    if not pad_image_flag:
        return image
    height, width, _ = image.shape
    long_side = max(width, height)
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    image_t[:, :] = [127, 127, 127]
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize(image, insize):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    return image

class MyPreproc(object):

    def __init__(self, img_dim, kpt_flip_map=None):
        self.img_dim = img_dim
        self.kpt_flip_map = kpt_flip_map

    def __call__(self, img, bboxes, keypoints, kpts_vis):
        assert bboxes.shape[0] > 0, "this image does not have gt"
        boxes = bboxes.copy()
        kpts_visibility = kpts_vis.copy()
        kpts = keypoints.copy()

        image_t, boxes_t, kpts_visibility_t, kpts_t, pad_image_flag = _crop(img, boxes, kpts_visibility, kpts, self.img_dim)
        image_t = _distort(image_t)
        image_t = _pad_to_square(image_t, pad_image_flag)
        image_t, boxes_t, kpts_t, kpts_visibility_t = _mirror(image_t, boxes_t, kpts_t, kpts_visibility_t, self.kpt_flip_map)
        height, width, _ = image_t.shape
        image_t = _resize(image_t, self.img_dim)
        
        # Normalize coordinates
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        kpts_t[:, 0::2] /= width
        kpts_t[:, 1::2] /= height

        return image_t, boxes_t, kpts_t, kpts_visibility_t
    
class preproc_val(object):

    def __init__(self, img_dim):
        self.img_dim = img_dim

    def __call__(self, img):
        return img