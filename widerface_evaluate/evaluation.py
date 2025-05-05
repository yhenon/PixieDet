"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import numpy as np
import os
import torch
from scipy.io import loadmat
import cv2
from widerface_evaluate.bbox import bbox_overlaps
from glob import glob
from torchvision.ops import nms
from anchors import generate_candidate_points
import torch.nn.functional as F


def _bbox_decode( priors, bbox_preds):
    xys = (bbox_preds[..., :2] * priors[..., 2:]) + priors[..., :2]
    whs = bbox_preds[..., 2:].exp() * priors[..., 2:]

    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)

    decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
    return decoded_bboxes


def get_val_preds(model, dataset_folder, confidence_threshold=0.02, nms_threshold=0.45, origin_size=False, device='cuda', save_image=False):

    # testing dataset
    save_folder = './widerface_evaluate/widerface_txt/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    filenames = glob(os.path.join(dataset_folder, 'val', 'images') + '/*/*.jpg')

    num_images = len(filenames)

    # testing begin
    for i, image_path in enumerate(filenames):
        img_name = os.path.basename(image_path)
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # pad to nearest multiple of 32
        imgp = np.zeros(((img_raw.shape[0] + 31) // 32 * 32, (img_raw.shape[1] + 31) // 32 * 32, 3), dtype=np.uint8)
        imgp[:img_raw.shape[0], :img_raw.shape[1], :] = img_raw
        img_raw = imgp

        #img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        if not origin_size:
            height, width, _ = img_raw.shape

            input_size = (640, 480)
            assert len(input_size) == 2
            x, y = max(input_size), min(input_size)
            if img_raw.shape[1] > img_raw.shape[0]:
                input_size = (x, y)
            else:
                input_size = (y, x)
            im_ratio = float(img_raw.shape[0]) / img_raw.shape[1]
            model_ratio = float(input_size[1]) / input_size[0]
            if im_ratio > model_ratio:
                new_height = input_size[1]
                new_width = int(new_height / im_ratio)
            else:
                new_width = input_size[0]
                new_height = int(new_width * im_ratio)

            img = cv2.resize(img_raw, (new_width, new_height), fx=None, fy=None, interpolation=cv2.INTER_LINEAR)
            resize_scale = torch.from_numpy(np.array([scale_x, scale_y, scale_x, scale_y])).to(device)
        else:
            img = img_raw.copy()
            resize_scale = 1

        #img = np.float32(img)  / 255 - 0.5
        img = np.float32(img)  

        im_height, im_width, _ = img.shape
        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)  # to (B,C,H,W)
        img = img.to(device)
        centers, strides = generate_candidate_points((im_height, im_width), device=device, add_half_stride=False)

        # scale centers by image size to normalize into [0,1]
        size = torch.Tensor([im_width, im_height]).to(device)
        centers = centers / size

        # scale strides based on training
        #strides = strides / 320
        strides = strides.to(device)
        
        obj_out, bbox_out, cls_out, kpts_out = model(img)
        conf = obj_out.sigmoid() * cls_out.sigmoid()
        #import pdb;pdb.set_trace()
        # decode raw bbox outputs into absolute coordinates:
        #   bbox_out: (B, N, 4) offsets in “stride” units
        #   strides:  (N,)
        #   centers:  (N,2)
        # 1) scale offsets by stride
        #scaled_offsets = bbox_out * strides.unsqueeze(0).unsqueeze(-1)
        #scaled_offsets[:,:, :2] *= -1

        # 2) add center coords to both sides of the box (x1,y1,x2,y2)
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        scale = scale.to(device)

        centers_expanded = centers.unsqueeze(0).expand(1, -1, -1)
        centers_scaled = centers_expanded*scale.unsqueeze(0)[:,:2]
        priors = torch.cat([centers_scaled[0,:,:], strides.unsqueeze(1), strides.unsqueeze(1)], dim=-1)
        boxes = _bbox_decode(priors, bbox_out)
        #boxes = scaled_offsets + torch.cat([centers_scaled, centers_scaled], dim=-1)
        boxes = (boxes * resize_scale).squeeze(0)
        
        scores = conf.squeeze(0).squeeze(1)

        inds = (scores > confidence_threshold)
        boxes = boxes[inds, :]
        #landms = landms[inds]
        scores = scores[inds]

        # do NMS
        keep = nms(boxes, scores, nms_threshold)

        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        #landms = landms[keep].cpu().numpy()

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        #dets = np.concatenate((dets, landms), axis=1)

        # --------------------------------------------------------------------
        save_name = os.path.join(save_folder, os.path.basename(os.path.dirname(image_path)), img_name[:-4] + ".txt")
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)

        if i % 1000 == 0:
            print(f'Evaluation images: {i:d}/{num_images:d}')

        # save image
        if save_image:
            if not origin_size:
                img_raw = cv2.resize(img_raw, None, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)

            for b in dets:
                if b[4] < 0.5:
                    continue

                text = f"{b[4]:.2f}"
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                #cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                #cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                #cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            if not os.path.exists("./results/"):
                os.makedirs("./results/")
            name = "./results/" + str(i) + ".jpg"
            #cv2.imwrite(name, img_raw)
            cv2.imshow('name', img_raw)
            cv2.waitKey(0)


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] == '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            current_event[imgname.rstrip('.jpg')] = _boxes
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """
    max_score = 0
    min_score = 1

    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            _min = np.min(v[:, -1])
            _max = np.max(v[:, -1])
            max_score = max(_max, max_score)
            min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
        for _, v in k.items():
            if len(v) == 0:
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    pred = get_preds(pred)
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num))
        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]

                gt_boxes = gt_bbx_list[j][0].astype('float')
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)

                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")
    print("Reference (pytorch retinaface mnet0.25, orig scale)	 90.7%	88.2%	73.8%    (Mean: 84.2%)")
    print("Reference (Yunet original, orig scale)	             89.2%  88.3%   81.1%    (Mean: 86.2%)")
    print("=================================================")

    return aps


