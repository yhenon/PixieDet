import cv2
import torch
import numpy as np


def vis(img_np, boxes, keypoints, idx):
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    h, w, _ = img_np.shape
    for i in range(boxes.shape[0]):
        bbox = [float(x) for x in boxes[i,:]]
        bbox[0] = int(bbox[0] * w)
        bbox[1] = int(bbox[1] * h)
        bbox[2] = int(bbox[2] * w)
        bbox[3] = int(bbox[3] * h)
        cv2.rectangle(img_np, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255))
    cv2.imwrite(f'out/x{idx}.jpg', img_np)

def visualize_assignment(img_tensor: torch.Tensor,
                         centers: torch.Tensor,
                         labels: torch.Tensor,
                         assigned_boxes: torch.Tensor,
                         gt_boxes_xywh: torch.Tensor,
                         kpts: torch.Tensor,
                         kpts_vis = None):
    """
    img_tensor: (C, H, W) float tensor in range [-0.5, 0.5]
    centers:    (N, 2) tensor of anchor centers (in pixels)
    labels:     (N,)  tensor of assignments (0=bg, 1..M=gt idx+1)
    assigned_boxes: (N, 4) tensor of matched gt boxes (x,y,w,h)
    gt_boxes_xywh:  (M, 4) original GT boxes for reference
    """
    #import pdb;pdb.set_trace()
    # 1) Bring image back to HxWx3 uint8 BGR
    img = img_tensor.detach().cpu().numpy()
    img = ((img + 0.) * 1).clip(0,255).astype(np.uint8)      # [0,255]
    #img = ((img + 0.5) * 255).clip(0,255).astype(np.uint8)      # [0,255]
    img = img.transpose(1,2,0).copy()                           # H,W,C RGB
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print(img.shape)
    H, W = img.shape[:2]
    print(kpts.shape)
    gt_boxes_xywh[:, 0] *= W
    gt_boxes_xywh[:, 1] *= H
    gt_boxes_xywh[:, 2] *= W
    gt_boxes_xywh[:, 3] *= H

    kpts[:, ::2] *= W
    kpts[:, 1::2] *= H

    # 2) Draw all GT boxes in green
    for (x, y, w, h) in gt_boxes_xywh.cpu().numpy():
        cx = x + w/2
        cy = y + h/2
        x1 = int(cx - w/2);  y1 = int(cy - h/2)
        x2 = int(cx + w/2);  y2 = int(cy + h/2)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (127, 0, 127), (0, 127, 127)]
    for ix in range(kpts.shape[0]):
        for j in range(kpts.shape[1]//2):
            if kpts_vis is not None:
                if kpts_vis[ix, j * 2] == 0:
                    continue

            xx = kpts[ix, j * 2].item()
            yy = kpts[ix, j * 2 + 1].item()
            
            cv2.circle(img, (int(xx), int(yy)), 2, colors[j % len(colors)], -1)

    # 3) For each positive anchor, draw a small circle at its center,
    #    and its assigned box in red
    pos_idxs = torch.nonzero(labels >= 0, as_tuple=False).view(-1)
    for idx in pos_idxs.cpu().tolist():
        # center
        cx, cy = (W*centers[idx]).int().cpu().tolist()
        
        cv2.circle(img, (cx, cy), radius=3, color=(0,0,255), thickness=-1)
        # assigned box
        #ax, ay, aw, ah = assigned_boxes[idx].cpu().numpy()*W
        #x1 = int(ax);  y1 = int(ay)
        #x2 = int(ax + aw);  y2 = int(ay + ah)
        #cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)
    # 4) Show
    cv2.imshow(f"imf", img)
    cv2.waitKey(0)

