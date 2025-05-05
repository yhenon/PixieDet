import torch

def collate_data(input):
    imgs = []
    bboxes = []
    kpts = []
    kpts_vis = []
    for img, bbox, kpt, kpt_vis in input:
        bboxes.append(bbox)
        kpts.append(torch.from_numpy(kpt))
        kpts_vis.append(torch.from_numpy(kpt_vis))
        imgs.append(torch.from_numpy(img).permute(2,0,1).unsqueeze(0))
    imgs = torch.cat(imgs,dim=0).float()


    return imgs, bboxes, kpts, kpts_vis