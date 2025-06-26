import torch
from bbox import BBoxBatch
from atss import compute_iou


def test_compute_iou_identity():
    box = torch.tensor([[0., 0., 1., 1.]])
    result = compute_iou(box, box)
    assert torch.allclose(result, torch.tensor([[1.0]]))


def test_compute_iou_no_overlap():
    box1 = torch.tensor([[0., 0., 1., 1.]])
    box2 = torch.tensor([[2., 2., 3., 3.]])
    result = compute_iou(box1, box2)
    assert torch.allclose(result, torch.tensor([[0.0]]))


def test_bboxbatch_iou_matches_compute_iou():
    boxes1 = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.]])
    boxes2 = torch.tensor([[0., 0., 1., 1.], [2., 2., 4., 4.]])

    expected = compute_iou(boxes1, boxes2)

    bb1 = BBoxBatch(boxes1.unsqueeze(0))
    bb2 = BBoxBatch(boxes2.unsqueeze(0))
    result = bb1.iou(bb2)[0]
    assert torch.allclose(result, expected)


def test_compute_iou_half_overlap():
    box1 = torch.tensor([[0., 0., 1., 1.]])
    box2 = torch.tensor([[0., 0., 1., 2.]])
    result = compute_iou(box1, box2)
    assert torch.allclose(result, torch.tensor([[0.5]]))
