import math

def compute_iou(pred, target, num_classes):
    iou_list = []
    pred = pred.view(-1)
    target = target.view(-1)

    # For classes excluding the background
    for cls in range(num_classes - 1):  # We subtract 1 to exclude the background class
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum().float()
        union = (pred_inds + target_inds).sum().float()
        if union == 0:
            iou_list.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            iou_list.append((intersection / union).item())
    return iou_list

def compute_mIoU(preds, labels, num_classes=13):
    iou_list = compute_iou(preds, labels, num_classes)
    valid_iou_list = [iou for iou in iou_list if not math.isnan(iou)]
    mIoU = sum(valid_iou_list) / len(valid_iou_list)
    return mIoU