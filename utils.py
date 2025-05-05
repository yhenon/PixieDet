import torch
def _kpt_decode(priors, deltas):
    # example shapes
    N, K = priors.shape[0], deltas.shape[1] // 2

    # unpack priors
    # cx, cy, stride_x, stride_y all [N]
    cx, cy, stride_x, stride_y = priors.unbind(dim=1)

    # reshape deltas to [N, K, 2]
    offsets = deltas.view(N, K, 2)

    # now decode:
    #   x_decoded = cx[..., None] + offsets[...,0] * stride_x[..., None]
    #   y_decoded = cy[..., None] + offsets[...,1] * stride_y[..., None]
    x = cx.unsqueeze(1) + offsets[..., 0] * stride_x.unsqueeze(1)
    y = cy.unsqueeze(1) + offsets[..., 1] * stride_y.unsqueeze(1)

    # stack back to [N, K, 2]
    keypoints = torch.stack((x, y), dim=-1)

    # if you really want [N, 2*K] again:
    return keypoints.view(N, 2*K)

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, scheduler, epoch, loss
