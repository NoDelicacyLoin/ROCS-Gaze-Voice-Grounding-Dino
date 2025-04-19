import torch
import torch.nn.functional as F
from torch import nn

class SetCriterion(nn.Module):
    def __init__(self, weight_dict):
        super().__init__()
        self.weight_dict = weight_dict

    def loss_labels(self, outputs, targets):
        src_logits = outputs["pred_logits"]  # [B, num_queries, num_classes]
        target_classes = torch.cat([t["labels"] for t in targets])
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes.unsqueeze(0))

    def loss_boxes(self, outputs, targets):
        src_boxes = outputs["pred_boxes"]
        target_boxes = torch.cat([t["boxes"] for t in targets])
        return F.l1_loss(src_boxes, target_boxes.unsqueeze(0))

    def loss_gaze(self, outputs, targets):
        if "gaze_attn" not in outputs:
            return torch.tensor(0.0, device=outputs["pred_boxes"].device)

        gaze_preds = outputs["gaze_attn"]  # [B, 1, H, W] or [B, H, W]
        gaze_targets = [t["heatmap"].to(gaze_preds.device) for t in targets]
        loss = 0.0
        for pred, gt in zip(gaze_preds, gaze_targets):
            if pred.dim() == 3:  # [1, H, W]
                pred = pred.squeeze(0)
            loss += F.mse_loss(pred, gt)
        return loss / len(gaze_preds)

    def forward(self, outputs, targets):
        losses = {}

        # 标签分类损失
        losses["loss_ce"] = self.loss_labels(outputs, targets) * self.weight_dict.get("loss_ce", 1.0)

        # 边框回归损失
        losses["loss_bbox"] = self.loss_boxes(outputs, targets) * self.weight_dict.get("loss_bbox", 5.0)

        # 眼动特征损失
        losses["loss_gaze"] = self.loss_gaze(outputs, targets) * self.weight_dict.get("loss_gaze", 1.0)

        return losses
