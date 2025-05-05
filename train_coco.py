import os
import argparse
from collections import deque
import time
import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from functools import partial
from models.yunet import Yunet
from coco import COCOKeypointDataset
from anchors import generate_candidate_points
from collate import collate_data
from vis import visualize_assignment
from simota import SimOTAAssigner
from losses import compute_losses_per_image
from bbox import BBoxBatch, _bbox_decode
from utils import _kpt_decode, save_checkpoint, load_checkpoint
from data_augment import MyPreproc


def train_model(
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,
    target_size=320,
    log_period=10,
    data_dir=None,
    checkpoint_dir="checkpoints",
    predictions_dir="predictions",
    device="cuda",
    eval_frequency=5,  # Evaluate every N epochs
    deque_size=100,    # Size of the sliding window for metrics
    freeze_backbone=False,
    resume=True,
    swa_start=0.75,    # Start SWA at 75% of training
    swa_lr=1e-4,       # SWA learning rate (10x smaller than initial)
    swa_anneal_epochs=10,  # Number of epochs to anneal from current LR to SWA LR
    swa_freq=1         # Update SWA model every N epochs
):

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Set up dataset paths
    train_root = os.path.join(data_dir, "train")
    train_label_file = os.path.join(train_root, "labelv2.txt")
    
    # Initialize model
    model = Yunet(num_kpts=17).to(device)
    

    # Calculate SWA start epoch based on percentage
    swa_start_epoch = int(num_epochs * swa_start)
    print(f"SWA will start at epoch {swa_start_epoch} ({swa_start*100:.0f}% of training)")
    
    # Initialize SWA model
    swa_model = AveragedModel(model)
    
    # Data augmentation
    kpt_flip_map = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    train_augmenter = MyPreproc(target_size, kpt_flip_map=kpt_flip_map)
    
    # Dataset and dataloader
    dataset = COCOKeypointDataset(
        json_path="/home/yannh/datasets/coco/annotations/person_keypoints_train2017.json", 
        img_dir="/home/yannh/datasets/coco/train2017/",
        preproc=train_augmenter
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_data, 
        drop_last=True, 
        num_workers=8
    )
    print(f"Number of images: {len(dataset)}")
    
    # Generate candidate points
    centers, strides = generate_candidate_points((target_size, target_size), device=device, add_half_stride=False)
    centers /= target_size
    assert centers.shape[0] == strides.shape[0]
    assigner = SimOTAAssigner()
    
    # Optimizer
    if freeze_backbone:
        print("Freezing backbone, only training detection heads")
        for p in model.parameters():
            p.requires_grad_ = False
        for p in model.bbox_head.multi_level_cls.parameters():
            p.requires_grad_ = True
        for p in model.bbox_head.multi_level_kps.parameters():
            p.requires_grad_ = True

    #optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad_], lr=learning_rate, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad_], lr=learning_rate, weight_decay=weight_decay)

    steps_per_epoch = len(dataset) // batch_size + 1  # Number of batches per epoch

    # Learning rate scheduler for pre-SWA phase
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=swa_start_epoch * steps_per_epoch,  # Only schedule until SWA starts
        pct_start=0.3,  # Spend 30% of training time in the increasing phase
        anneal_strategy='cos',  # Use cosine annealing
        div_factor=25,   # Initial LR = max_lr/div_factor
        final_div_factor=1000,  # Final LR = max_lr/final_div_factor
        three_phase=False  # Use two-phase (increase, then decrease) schedule
    )

    # Try to load latest checkpoint if exists and resume flag is set
    if resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if not latest_checkpoint:
            raise ValueError('No checkpoint found')
        print(f"Loading checkpoint from {latest_checkpoint}")
        model, optimizer, scheduler, start_epoch, loss = load_checkpoint(latest_checkpoint, model, optimizer, scheduler)
        print(f"Resuming training from epoch {start_epoch} with loss {loss}")
    else:
        start_epoch = 0

    # Initialize sliding window for metrics
    losses_obj = deque(maxlen=deque_size)
    losses_iou = deque(maxlen=deque_size)
    losses_bbox = deque(maxlen=deque_size)
    losses_kpt = deque(maxlen=deque_size)
    losses_total = deque(maxlen=deque_size)
    #model.load_state_dict(torch.load('model_epoch_16.pth'))#['model_state_dict'])
    # Training loop
    best_map = 0.0
    priors_cat = torch.cat([(centers)*target_size + (strides/2).unsqueeze(1).repeat(1,2), strides.unsqueeze(1), strides.unsqueeze(1)],dim=1)
    anchors = torch.cat([centers, (strides / target_size).unsqueeze(1), (strides / target_size).unsqueeze(1)],dim=1)

    for epoch_num in range(start_epoch, num_epochs):
        model.train()
        epoch_start_time = time.time()
        
        # Check if we're entering SWA phase
        is_swa_phase = epoch_num >= swa_start_epoch
        
        # If we're entering SWA phase, switch to constant or cyclical LR
        if is_swa_phase and epoch_num == swa_start_epoch:
            print("\n==== Entering SWA Phase ====")
            print(f"Switching to SWA learning rate: {swa_lr}")
            
            # If we want to gradually anneal to SWA LR
            if swa_anneal_epochs > 0:
                # Get current LR
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Annealing from current LR {current_lr} to SWA LR {swa_lr} over {swa_anneal_epochs} epochs")
                
                # Create linear scheduler to anneal from current LR to SWA LR
                steps_per_anneal = swa_anneal_epochs * steps_per_epoch
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=current_lr / swa_lr,
                    end_factor=1.0,
                    total_iters=steps_per_anneal
                )
            else:
                # Just set the LR directly to SWA LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = swa_lr
                
                # Use constant LR scheduler for SWA phase
                scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        
        # Initialize epoch metrics
        epoch_obj_loss = 0.0
        epoch_iou_loss = 0.0
        epoch_bbox_loss = 0.0
        epoch_kpt_loss = 0.0
        epoch_total_loss = 0.0
        
        for iter_num, (imgs, boxes_gt, kpts, kpts_vis) in enumerate(dataloader):

            imgs = imgs.to(device)
            boxes_gt = [b.to(device) for b in boxes_gt]
            kpts = [b.to(device) for b in kpts]
            kpts_vis = [b.to(device) for b in kpts_vis]
            
            # Forward pass
            obj_out, bbox_out, cls_out, kpts_out = model(imgs)
            
            # Calculate losses
            total_obj = total_iou = total_bbox = total_kpt = total_loss = 0.0
            
            bbox_pred = _bbox_decode(anchors.unsqueeze(0).repeat(batch_size, 1, 1), bbox_out)
            bbox_pred = BBoxBatch(bbox_pred, fmt="xyxy", image_sizes=imgs.shape[2:])
            conf = obj_out.sigmoid()*cls_out.sigmoid() # (B, N, 1)
            conf = torch.cat([1-conf, conf],dim=2) # (B, N, 2)

            # Calculate loss for each image in batch
            for b in range(imgs.size(0)):
                num_gt, labels, max_overlaps, assigned_labels = assigner.assign(
                    conf[b, :, :],
                    priors_cat,
                    bbox_pred.get_b(b).boxes.squeeze(0) * target_size,
                    boxes_gt[b]._ensure_xyxy().squeeze(0)*target_size,
                    torch.ones(boxes_gt[b].boxes.shape[1])
                )
                labels -= 1

                l_obj, l_iou, l_box = compute_losses_per_image(
                    cls_out[b],                  # (N,)
                    obj_out[b].squeeze(-1),      # (N,)
                    bbox_pred.get_b(b),          # (N,4)
                    labels,                      # (N, )
                    boxes_gt[b].to(imgs.device)  # (M, 4)
                )
                kpt_target = (kpts[b]*target_size)[labels[labels>=0]]
                kpt_pred = _kpt_decode(priors_cat, kpts_out[b])[labels>=0]
                kpt_vis_mask = (kpts_vis[b])[labels[labels>=0]]
                if kpt_vis_mask.sum() > 0:
                    l_kpt = (F.smooth_l1_loss(kpt_target, kpt_pred, reduction='none') * kpt_vis_mask).mean() / 10
                else:
                    l_kpt = torch.tensor(0.0, device=device)

                total_obj += l_obj / imgs.size(0)
                total_iou += l_iou / imgs.size(0)
                total_bbox += l_box / imgs.size(0)
                total_kpt += l_kpt / imgs.size(0)
                # visualize_assignment(imgs[b,:,:,:], centers  + (strides/2/target_size).unsqueeze(1).repeat(1,2), labels, None, boxes_gt[b].boxes[0,:,:], kpts[b], kpts_vis[b])

            # Total loss
            loss = total_obj + total_iou + total_bbox + 1 * total_kpt
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            #scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            obj_loss = total_obj.item()
            iou_loss = total_iou.item()
            bbox_loss = total_bbox.item()
            kpt_loss = total_kpt.item()
            total_loss = loss.item()
            
            # Add to sliding window
            losses_obj.append(obj_loss)
            losses_iou.append(iou_loss)
            losses_bbox.append(bbox_loss)
            losses_kpt.append(kpt_loss)
            losses_total.append(total_loss)
            
            # Update epoch metrics
            epoch_obj_loss += obj_loss
            epoch_iou_loss += iou_loss
            epoch_bbox_loss += bbox_loss
            epoch_kpt_loss += kpt_loss
            epoch_total_loss += total_loss
                        
            # Log metrics periodically
            if (iter_num + 1) % log_period == 0:
                window_obj = sum(losses_obj) / len(losses_obj)
                window_iou = sum(losses_iou) / len(losses_iou)
                window_bbox = sum(losses_bbox) / len(losses_bbox)
                window_kpt = sum(losses_kpt) / len(losses_kpt)
                window_total = sum(losses_total) / len(losses_total)
                
                print(f"Iter {iter_num+1:4d}/{len(dataloader)} | "
                      f"Window {min(deque_size, len(losses_total))} iters: "
                      f"obj={window_obj:.4f}, iou={window_iou:.4f}, kpt={window_kpt:.4f} "
                      f"bbox={window_bbox:.4f}, total={window_total:.4f} | "
                      f"Current LR: {optimizer.param_groups[0]['lr']:.6f}" +
                      (" [SWA]" if is_swa_phase else ""))
        
        # Calculate epoch averages
        epoch_obj_loss /= len(dataloader)
        epoch_iou_loss /= len(dataloader)
        epoch_bbox_loss /= len(dataloader)
        epoch_kpt_loss /= len(dataloader)        
        epoch_total_loss /= len(dataloader)
        
        # Update SWA model if in SWA phase and frequency matches
        if is_swa_phase and (epoch_num - swa_start_epoch) % swa_freq == 0:
            swa_model.update_parameters(model)
            print(f"Updated SWA model (updates: {(epoch_num - swa_start_epoch) // swa_freq + 1})")
        
        # Log epoch metrics
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch_num}/{num_epochs - 1} completed in {epoch_time:.2f}s")
        print(f"Epoch Metrics:")
        print(f"  Objectness Loss: {epoch_obj_loss:.4f}")
        print(f"  IOU-branch Loss: {epoch_iou_loss:.4f}")
        print(f"  Bounding Box Loss: {epoch_bbox_loss:.4f}")
        print(f"  Kpt Loss: {epoch_kpt_loss:.4f}")
        print(f"  Total Loss: {epoch_total_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint of regular model
        save_checkpoint(model, optimizer, scheduler, epoch_num, epoch_total_loss, f'{checkpoint_dir}/coco_model_epoch_{epoch_num}.pth')
        
        # Evaluate on validation set periodically
        if (epoch_num % eval_frequency == 0 and epoch_num >= 20):
            # First evaluate the regular model
            print(f"\nEvaluating regular model on validation set...")
            mean_ap = evaluate_model(model, data_dir)
            print(f"Regular Model Validation mAP: {mean_ap:.4f}")
            
            # Save best model
            if mean_ap > best_map:
                best_map = mean_ap
                torch.save(model.state_dict(), f'{checkpoint_dir}/model_best.pth')
                print(f"New best model saved with mAP: {best_map:.4f}")
            
            # If in SWA phase, also evaluate the SWA model
            if is_swa_phase:
                print(f"\nEvaluating SWA model on validation set...")
                # Get a temporary copy of the SWA model for evaluation
                swa_model_eval = AveragedModel(model)
                swa_model_eval.load_state_dict(swa_model.state_dict())
                
                # Update batch norm statistics
                print("Updating batch norm statistics for SWA model...")
                update_bn(dataloader, swa_model_eval, device=device)
                
                # Evaluate SWA model
                swa_mean_ap = evaluate_model(swa_model_eval, data_dir)
                print(f"SWA Model Validation mAP: {swa_mean_ap:.4f}")
                
                # Save best SWA model
                if swa_mean_ap > best_map:
                    best_map = swa_mean_ap
                    torch.save(swa_model_eval.state_dict(), f'{checkpoint_dir}/swa_model_best.pth')
                    print(f"New best model (SWA) saved with mAP: {best_map:.4f}")
                
                # Also save the current SWA model
                torch.save(swa_model_eval.state_dict(), f'{checkpoint_dir}/swa_model_epoch_{epoch_num}.pth')
    
    # End of training - finalize the SWA model
    if swa_start_epoch < num_epochs:
        print("\n==== Finalizing SWA Model ====")
        
        # Update batch norm statistics for the final SWA model
        print("Updating batch norm statistics...")
        update_bn(dataloader, swa_model, device=device)
        
        # Save the final SWA model
        torch.save(swa_model.state_dict(), f'{checkpoint_dir}/swa_model_final.pth')
        
        # Final evaluation of SWA model
        print("Evaluating final SWA model...")
        final_swa_map = evaluate_model(swa_model, data_dir)
        print(f"Final SWA Model Validation mAP: {final_swa_map:.4f}")
        
        # Save as best if it's better
        if final_swa_map > best_map:
            best_map = final_swa_map
            torch.save(swa_model.state_dict(), f'{checkpoint_dir}/model_best.pth')
            print(f"New best model (final SWA) saved with mAP: {best_map:.4f}")
    
    print(f"\nTraining completed. Best validation mAP: {best_map:.4f}")


def evaluate_model(model, data_dir):
    """Evaluate model on validation set"""
    model.eval()
    with torch.no_grad():
        get_val_preds(
            model=model, 
            dataset_folder=data_dir, 
            origin_size=True, 
            save_image=False
        )
    
    # Get average precision values
    aps = evaluation(
        pred='widerface_evaluate/widerface_txt/', 
        gt_path='widerface_evaluate/ground_truth/'
    )
    model.train()
    return sum(aps) / len(aps) if aps else 0.0


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('model_epoch_') and f.endswith('.pth')]
    if not checkpoints:
        return None
    
    # Extract epoch numbers and find the latest
    latest = None
    latest_epoch = -1
    for ckpt in checkpoints:
        try:
            epoch = int(ckpt.split('_')[-1].split('.')[0])
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest = ckpt
        except:
            continue
    
    return os.path.join(checkpoint_dir, latest) if latest else None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="YuNet Face Detection Training with SWA")
    
    # Dataset parameters
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Base directory containing WIDER dataset (with train/ and val/ subdirectories)")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=1000,
                        help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-2,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--target_size", type=int, default=640,
                        help="Target image size for training")
    
    # Model parameters
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone and only train detection heads")
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--predictions_dir", type=str, default="predictions",
                        help="Directory to save validation predictions")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from latest checkpoint if available")
    
    # SWA parameters
    parser.add_argument("--swa_start", type=float, default=0.75,
                        help="SWA start epoch as a fraction of total epochs (e.g., 0.75 means 75% of the way through training)")
    parser.add_argument("--swa_lr", type=float, default=1e-4,
                        help="SWA learning rate (typically ~10x smaller than initial LR)")
    parser.add_argument("--swa_anneal_epochs", type=int, default=10,
                        help="Number of epochs to anneal from current LR to SWA LR (0 means immediate switch)")
    parser.add_argument("--swa_freq", type=int, default=1,
                        help="Frequency to update SWA model (in epochs)")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--log_period", type=int, default=10,
                        help="Print log info every N iterations")
    parser.add_argument("--eval_frequency", type=int, default=5,
                        help="Evaluate model every N epochs")
    parser.add_argument("--deque_size", type=int, default=100,
                        help="Size of sliding window for metrics averaging")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(12)
    torch.cuda.manual_seed(12)
    
    # Print training parameters
    print(f"Starting training with parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Train model with arguments
    train_model(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_size=args.target_size,
        log_period=args.log_period,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        predictions_dir=args.predictions_dir,
        device=args.device,
        eval_frequency=args.eval_frequency,
        deque_size=args.deque_size,
        freeze_backbone=args.freeze_backbone,
        resume=args.resume,
        swa_start=args.swa_start,
        swa_lr=args.swa_lr,
        swa_anneal_epochs=args.swa_anneal_epochs,
        swa_freq=args.swa_freq
    )
