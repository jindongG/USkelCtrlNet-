# loss_functions.py
import torch
import torch.nn.functional as F


# ------------------------
# Basic Dice
# ------------------------
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred and target expected to be probabilities in [0,1]
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=[1, 2, 3])
        denominator = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss.mean()


# ------------------------
# clDice with soft-skeleton approximation
# ------------------------
class clDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, sk_iter=8):
        super(clDiceLoss, self).__init__()
        self.smooth = smooth
        self.sk_iter = sk_iter

    def soft_erode(self, x, kernel_size=3):
        # min-pool via -max-pool on negated tensor (differentiable approx)
        return -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size // 2)

    def soft_skeleton(self, x, iters=None):
        # iterative thinning-like approximation (returns soft skeleton)
        if iters is None:
            iters = self.sk_iter
        curr = x
        skeleton = torch.zeros_like(x)
        for _ in range(iters):
            eroded = self.soft_erode(curr, 3)
            opened = F.max_pool2d(eroded, 3, stride=1, padding=1)  # opening approx
            delta = (curr - opened).clamp(min=0.0)
            skeleton = torch.clamp(skeleton + delta, 0.0, 1.0)
            curr = eroded
        return skeleton

    def norm_intersection(self, center_line, vessel, eps=1.0):
        # center_line, vessel: (B,C,H,W)
        clf = center_line.view(*center_line.shape[:2], -1)
        vf = vessel.view(*vessel.shape[:2], -1)
        intersection = (clf * vf).sum(-1)
        return (intersection + eps) / (clf.sum(-1) + eps)

    def soft_cldice_loss(self, pred, target):
        # pred and target are probabilities in [0,1]
        spred = self.soft_skeleton(pred, iters=self.sk_iter)
        starget = self.soft_skeleton(target, iters=self.sk_iter)
        iflat = self.norm_intersection(spred, target)
        tflat = self.norm_intersection(starget, pred)
        inter = (iflat * tflat).sum(-1)
        denom = (iflat + tflat).sum(-1) + 1e-6
        cldice = 1.0 - (2.0 * inter) / denom
        return cldice.mean()

    def dice_loss(self, pred, target):
        # batch-wise dice
        intersection = (pred * target).sum(dim=[1, 2, 3])
        denominator = pred.sum(dim=[1, 2, 3]) + target.sum(dim=[1, 2, 3])
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss.mean()

    def forward(self, pred, target):
        # Weighted combination as in original paper/version: 0.8 dice + 0.2 clDice
        return 0.8 * self.dice_loss(pred, target) + 0.2 * self.soft_cldice_loss(pred, target)


# ------------------------
# Edge loss (Sobel guidance). We derive predicted edges from pred (no extra head required).
# ------------------------
def sobel_edge(img):
    # img: (B,1,H,W)
    device = img.device
    kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(img, kernel_x, padding=1)
    gy = F.conv2d(img, kernel_y, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
    # normalize to [0,1]
    grad = grad / (grad.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)
    return grad


def edge_bce_between_pred_and_gt(pred_prob, gt_mask, edge_thresh=0.05):
    # pred_prob: predicted probability map (B,1,H,W)
    # gt_mask: ground-truth binary mask (B,1,H,W) or float
    pred_edge = sobel_edge(pred_prob)  # derived predicted edge magnitude
    gt_edge = sobel_edge(gt_mask)
    # binarize gt edge with small threshold
    gt_edge_bin = (gt_edge > edge_thresh).float()
    # use BCE with logits? pred_edge is in [0,1], so use BCE (apply small clamp)
    pred_edge_clamped = pred_edge.clamp(min=1e-6, max=1 - 1e-6)
    bce = F.binary_cross_entropy(pred_edge_clamped, gt_edge_bin, reduction='mean')
    return bce


# ------------------------
# Connectivity proxy losses
#   - local_connectivity_loss: penalize high predicted pixel with low neighborhood mean
#   - small_component_loss: penalize small isolated components via opening approx
# ------------------------
def local_connectivity_loss(pred, kernel_size=7):
    # pred: (B,1,H,W) probabilities
    pad = kernel_size // 2
    device = pred.device
    weight = torch.ones((1, 1, kernel_size, kernel_size), device=device)
    sum_local = F.conv2d(pred, weight, padding=pad)
    mean_local = sum_local / (kernel_size * kernel_size)
    loss_map = pred * (1.0 - mean_local)
    return loss_map.mean()


def soft_erode(pred, kernel_size=3, iterations=1):
    # same as in clDice class but standalone
    x = pred
    for _ in range(iterations):
        x = -F.max_pool2d(-x, kernel_size, stride=1, padding=kernel_size // 2)
    return x


def small_component_loss(pred, kernel_size=3, iters=1):
    # opening (erode then dilate), penalize removed pixels (small components)
    eroded = soft_erode(pred, kernel_size=kernel_size, iterations=iters)
    # dilate (approx by -max_pool on negated)
    dilated = -F.max_pool2d(-eroded, kernel_size, stride=1, padding=kernel_size // 2)
    diff = (pred - dilated).clamp(min=0.0)
    return diff.mean()


def connectivity_loss(pred, lam_local=1.0, lam_small=0.5):
    return lam_local * local_connectivity_loss(pred) + lam_small * small_component_loss(pred)


# ------------------------
# Total loss composition
# ------------------------
class TotalVesselLoss(torch.nn.Module):
    """
    Combines Dice, clDice, edge loss, connectivity proxy.
    alpha * Dice + beta * clDice + gamma * EdgeBCE + delta * Connectivity
    """

    def __init__(self, alpha=0.5, beta=0.3, gamma=0.1, delta=0.1):
        super(TotalVesselLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dice = DiceLoss()
        self.cldice = clDiceLoss()

    def forward(self, pred_prob, gt_mask):
        """
        pred_prob: predicted probabilities in [0,1], shape (B,1,H,W)
        gt_mask: ground-truth binary mask (B,1,H,W) float {0,1}
        """
        # Ensure shapes and types
        pred_prob = pred_prob.float()
        gt_mask = gt_mask.float()
        Ldice = self.dice(pred_prob, gt_mask)
        Lcl = self.cldice(pred_prob, gt_mask)
        Ledge = edge_bce_between_pred_and_gt(pred_prob, gt_mask)
        Lconn = connectivity_loss(pred_prob)
        total = self.alpha * Ldice + self.beta * Lcl + self.gamma * Ledge + self.delta * Lconn
        return total, dict(dice=Ldice, cldice=Lcl, edge=Ledge, conn=Lconn)


# ------------------------
# OHEM utility (pixel-level)
# ------------------------
def pixel_ohem_loss_from_logits(pred_logits, gt_mask, top_percent=0.3):
    """
    pred_logits: (B,1,H,W) logits (before sigmoid)
    gt_mask: (B,1,H,W) binary float
    Returns mean BCE over top-k hardest pixels per image.
    """
    pred_prob = torch.sigmoid(pred_logits)
    bce_per_pixel = F.binary_cross_entropy(pred_prob, gt_mask, reduction='none')  # (B,1,H,W)
    b, _, h, w = bce_per_pixel.shape
    flat = bce_per_pixel.view(b, -1)
    k = max(1, int(top_percent * flat.size(1)))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    return topk_vals.mean()


def pixel_ohem_loss_from_prob(pred_prob, gt_mask, top_percent=0.3):
    bce_per_pixel = F.binary_cross_entropy(pred_prob, gt_mask, reduction='none')  # (B,1,H,W)
    b, _, h, w = bce_per_pixel.shape
    flat = bce_per_pixel.view(b, -1)
    k = max(1, int(top_percent * flat.size(1)))
    topk_vals, _ = torch.topk(flat, k, dim=1)
    return topk_vals.mean()

