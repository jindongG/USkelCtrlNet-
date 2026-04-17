import os
import glob
from datetime import datetime
import cv2
import re
from scipy import ndimage

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_fill_holes
from skimage.morphology import skeletonize, label as sk_label

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

# ===== 你的工程依赖 =====
from dataset import DataLoader_Producer
from models.SwinSnake import SwinSnake_Alter
#from models.SwinSnake import SwinSnake_Dual, DSCNet
from monai.networks.nets import UNet, SegResNet, SwinUNETR, FlexUNet

try:
    from monai.networks.nets import DiNTS
    from monai.networks.nets.dints import TopologyInstance
except Exception:
    DiNTS = None
    TopologyInstance = None

# ============================================================
# Surface 参数（保留）
# ============================================================
SURFACE_DICE_TOL = 1.0  # 单位：像素。常用 1~2 px

# ============================================================
# 配置（按你原始脚本）
# ============================================================
DEVICE_ID = "0"

FOV = "6M"
LABEL_TYPE = "FAZ"  # "LargeVessel", "FAZ", "Capillary","Artery", "Vein",
BATCH_SIZE = 1

SPLIT = "test"      # "test" / "val" / "train"
THR = 0.5

MODEL_NAME = "FlexUNet"  # SwinSnake_Dual,DSCNet,DiNTS,SegResNet,UNet,FlexUNet,SwinUNETR,SwinSnake_Alter

HP = dict(
    layer_depth=3,
    kernel_size=15,
    extend_scope=1,
    down_layer="MaxPooling",
    rate=72,
    repeat_n=1,
)

CKPT_DIR = "/home/dataset-assist-0gjd/output/111/results/2026-02-03-04-59-26/FlexUNet_3_15_1_72_MaxPooling_1_True_6M_FAZ_100_#/checkpoints"
OUT_ROOT = "/home/dataset-assist-0gjd/Snake-Swin-OCTA-master(1)/results/6M"

PATTERN = "*.pth"
RECURSIVE = False

SAVE_PER_SAMPLE = True
MIN_MATCH_RATIO = 0.99

# ============================================================
# 保存预测（.npy）与可视化（彩色overlay + 白底黑前景）
# ============================================================
SKIP_VIS_FOR_FIRST_CKPT = True  # 第一个ckpt不做保存/可视化

ALPHA = 0.5
overlay = lambda x, y: cv2.addWeighted(x, ALPHA, y, 1 - ALPHA, 0)

# ===== 第二段的配色工具 =====
to_blue = lambda x: np.array([x, np.zeros_like(x), np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_red = lambda x: np.array([np.zeros_like(x), np.zeros_like(x), x]).transpose((1, 2, 0)).astype(np.uint8)
to_green = lambda x: np.array([np.zeros_like(x), x, np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_light_green = lambda x: np.array([np.zeros_like(x), x / 2, np.zeros_like(x)]).transpose((1, 2, 0)).astype(np.uint8)
to_yellow = lambda x: np.array([np.zeros_like(x), x, x]).transpose((1, 2, 0)).astype(np.uint8)

to_3ch = lambda x: np.array([x, x, x]).transpose((1, 2, 0)).astype(np.uint8)

def _safe_name(s: str) -> str:
    s = str(s)
    return re.sub(r"[^\w\-.]+", "_", s)

def remove_tiny_pieces(label_u8, min_area=10):
    """
    复用第二段可视化逻辑：输入 uint8(0/255) mask，移除小连通域（min_area=10）
    注意：这不是“50像素过滤”，这里保留仅用于可视化更干净。
    """
    structure = ndimage.generate_binary_structure(2, 2)
    labelmaps, connected_num = ndimage.label(label_u8 > 0, structure=structure)

    if connected_num == 0:
        return np.zeros_like(label_u8, dtype=np.uint8)

    component_sizes = np.bincount(labelmaps.ravel())[1:]
    tiny_components_labels = np.where(component_sizes < min_area)[0] + 1

    for lab in tiny_components_labels:
        labelmaps[labelmaps == lab] = 0

    out = (labelmaps > 0).astype(np.uint8) * 255
    return out

def tensor_image_to_u8_3ch(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [C,H,W] torch tensor
    输出: [H,W,3] uint8
    """
    img = img_t.detach().cpu().numpy()
    if img.ndim == 2:
        img = img[None, ...]

    mx = float(img.max()) if img.size else 1.0
    if mx <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[0] == 1:
        img3 = np.repeat(img, 3, axis=0)
    else:
        img3 = img[:3, ...]
    return img3.transpose(1, 2, 0)

def get_pred_colorizer(label_type: str):
    m = {
        "Vein": to_blue,
        "Artery": to_red,
        "Capillary": to_yellow,
        "RV": to_green,
        "LargeVessel": to_yellow,
        "FAZ": to_green,  # overlay里黑色不明显，这里用绿更直观
    }
    return m.get(label_type, to_yellow)

def save_pred_artifacts(pred_bool: np.ndarray, image_u8_3ch: np.ndarray, save_dir: str, base_name: str):
    """
    pred_bool: HxW bool
    image_u8_3ch: HxWx3 uint8
    保存：
      1) 预测mask .npy：0/255
      2) 可视化1：彩色overlay（第二段风格）
      3) 可视化2：白底黑前景（背景255，前景0）
    """
    os.makedirs(save_dir, exist_ok=True)

    pred_u8 = (pred_bool.astype(np.uint8) * 255)

    # 1) 保存 .npy
    np.save(os.path.join(save_dir, f"{base_name}.npy"), pred_u8)

    # 2) 可视化1：第二段风格 overlay
    pred_clean = remove_tiny_pieces(pred_u8, min_area=10)
    colorize = get_pred_colorizer(LABEL_TYPE)
    pred_color = colorize(pred_clean)
    vis_overlay = overlay(image_u8_3ch, pred_color)
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_overlay.png"), vis_overlay)

    # 3) 可视化2：白底黑前景
    wb = 255 - pred_u8
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_wb.png"), wb)

# ============================================================
# 模型构建
# ============================================================
class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        return self.new_layer(self.original_model(x))

def build_model(model_name: str, device, hp: dict):
    if model_name == "SwinSnake_Alter":
        return SwinSnake_Alter(
            img_ch=3, output_ch=1,
            layer_depth=hp["layer_depth"],
            kernel_size=hp["kernel_size"],
            extend_scope=hp["extend_scope"],
            down_layer=hp["down_layer"],
            rate=hp["rate"],
            repeat_n=hp["repeat_n"],
            device_id=DEVICE_ID
        ).to(device)




    if model_name == "SwinUNETR":
        model = SwinUNETR(in_channels=3, out_channels=1, feature_size=24 * hp["layer_depth"], spatial_dims=2)
        return ModifiedModel(model).to(device)

    if model_name == "UNet":
        N, B = 5, 8
        channels = [2 ** x for x in range(B, B + N)]
        strides = [2] * (len(channels) - 1)
        model = UNet(in_channels=3, out_channels=1, spatial_dims=2, channels=channels, strides=strides)
        return ModifiedModel(model).to(device)

    if model_name == "SegResNet":
        model = SegResNet(in_channels=3, out_channels=1, spatial_dims=2)
        return ModifiedModel(model).to(device)

    if model_name == "FlexUNet":
        model = FlexUNet(in_channels=3, out_channels=1, spatial_dims=2, backbone="efficientnet-b4")
        return ModifiedModel(model).to(device)

    if model_name == "DiNTS":
        if DiNTS is None or TopologyInstance is None:
            raise ImportError("DiNTS/TopologyInstance 导入失败，请按你本机 monai 版本修正 import。")
        dints_space = TopologyInstance(spatial_dims=2, device=f"cuda:{DEVICE_ID}")
        model = DiNTS(dints_space=dints_space, in_channels=3, num_classes=1, spatial_dims=2)
        return ModifiedModel(model).to(device)

    raise ValueError(f"Unsupported model: {model_name}")

def strip_module_prefix(sd: dict) -> dict:
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def load_state_shape_matched(model: nn.Module, sd: dict) -> float:
    msd = model.state_dict()
    filtered = {}
    for k, v in sd.items():
        if k in msd and hasattr(v, "shape") and hasattr(msd[k], "shape"):
            if tuple(v.shape) == tuple(msd[k].shape):
                filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    return len(filtered) / max(1, len(msd))

# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def ensure_prob(preds: torch.Tensor) -> torch.Tensor:
    if isinstance(preds, (tuple, list)):
        preds = preds[0]
    if preds.max() <= 1.0 and preds.min() >= 0.0:
        return preds.clamp(0.0, 1.0)
    return torch.sigmoid(preds)

def to_numpy(x):
    return x.detach().cpu().numpy()

def binarize(x, thr):
    return (x > thr).astype(np.bool_)

def confusion(pred, gt):
    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()
    return TP, FP, FN, TN

def dice(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    tot = pred.sum() + gt.sum()
    return (2 * inter) / (tot + eps)

def jaccard(pred, gt, eps=1e-6):
    inter = np.logical_and(pred, gt).sum()
    uni = np.logical_or(pred, gt).sum()
    return inter / (uni + eps)

def precision(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    return TP / (TP + FP + eps)

def recall(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    return TP / (TP + FN + eps)

def specificity(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    return TN / (TN + FP + eps)

def sens_spec(pred, gt, eps=1e-6):
    TP, FP, FN, TN = confusion(pred, gt)
    pos = TP + FN
    neg = TN + FP
    sens = TP / (pos + eps) if pos > 0 else 1.0
    spec = TN / (neg + eps) if neg > 0 else 1.0
    return float(sens), float(spec)

def bacc(pred, gt, eps=1e-6):
    sens, spec = sens_spec(pred, gt, eps=eps)
    return 0.5 * (sens + spec)

def gmean(pred, gt, eps=1e-6):
    sens, spec = sens_spec(pred, gt, eps=eps)
    return float(np.sqrt(max(sens * spec, 0.0)))

def hausdorff(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan
    p = np.argwhere(pred.astype(np.uint8))
    g = np.argwhere(gt.astype(np.uint8))
    d1 = directed_hausdorff(p, g)[0]
    d2 = directed_hausdorff(g, p)[0]
    return max(d1, d2)

def border(mask_bool):
    er = binary_erosion(mask_bool)
    return np.logical_xor(mask_bool, er)

def hd95(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    bp = border(pred)
    bg = border(gt)
    if bp.sum() == 0 or bg.sum() == 0:
        return np.nan

    dt_bg = distance_transform_edt(~bg)
    d_p2g = dt_bg[bp]
    dt_bp = distance_transform_edt(~bp)
    d_g2p = dt_bp[bg]

    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return np.nan

    return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

def surface_distances(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    bp = border(pred)
    bg = border(gt)

    if bp.sum() == 0 or bg.sum() == 0:
        return np.array([]), np.array([])

    dt_bg = distance_transform_edt(~bg)
    dt_bp = distance_transform_edt(~bp)

    d_p2g = dt_bg[bp]
    d_g2p = dt_bp[bg]
    return d_p2g, d_g2p

def assd(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    d_p2g, d_g2p = surface_distances(pred, gt)
    if d_p2g.size == 0 or d_g2p.size == 0:
        return np.nan
    return float(0.5 * (d_p2g.mean() + d_g2p.mean()))

def surface_dice(pred, gt, tol=1.0, eps=1e-6):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    bp = border(pred)
    bg = border(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0

    d_p2g, d_g2p = surface_distances(pred, gt)
    if d_p2g.size == 0 or d_g2p.size == 0:
        return 0.0

    m1 = (d_p2g <= tol).sum()
    m2 = (d_g2p <= tol).sum()
    return float((m1 + m2) / (bp.sum() + bg.sum() + eps))

def cldice_correct(pred, gt, eps=1e-6):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp = skeletonize(pred)
    sg = skeletonize(gt)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tprec = np.logical_and(sp, gt).sum() / (sp.sum() + eps)
    tsens = np.logical_and(sg, pred).sum() / (sg.sum() + eps)
    return (2 * tprec * tsens) / (tprec + tsens + eps)

def connectivity_lcc_ratio(pred, eps=1e-6):
    if pred.sum() == 0:
        return 0.0
    lab = sk_label(pred.astype(np.uint8), connectivity=2)
    counts = np.bincount(lab.flatten())
    if len(counts) <= 1:
        return 0.0
    counts = counts[1:]
    return float(counts.max() / (pred.sum() + eps))

def component_count(pred):
    lab = sk_label(pred.astype(np.uint8), connectivity=2)
    return int(lab.max())

def betti0_error(pred, gt):
    return abs(component_count(pred) - component_count(gt))

def count_holes(mask):
    filled = binary_fill_holes(mask.astype(bool))
    holes = np.logical_and(filled, ~mask.astype(bool))
    lab = sk_label(holes.astype(np.uint8), connectivity=2)
    return int(lab.max())

def betti1_error(pred, gt):
    return abs(count_holes(pred) - count_holes(gt))

def auc_roc(prob, gt):
    if roc_auc_score is None:
        return np.nan
    y_true = gt.astype(np.uint8).flatten()
    y_score = prob.astype(np.float32).flatten()
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan

METRICS = [
    "Dice", "Jaccard",
    "Precision", "Recall", "Specificity",
    "BACC", "GMean",
    "Hausdorff", "HD95",
    "ASSD", "SurfaceDice",
    "clDice", "AUC",
    "Connectivity", "ComponentCount",
    "Betti0Error", "Betti1Error",
]

def summarize(df):
    rows = []
    for m in METRICS:
        vals = df[m].to_numpy()
        vals = vals[~np.isnan(vals)]

        if len(vals) == 0:
            rows.append({
                "metric": m,
                "mean": np.nan, "std": np.nan,
                "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                "n_valid": 0,
            })
            continue

        q1 = float(np.percentile(vals, 25))
        q3 = float(np.percentile(vals, 75))
        rows.append({
            "metric": m,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "q1": q1,
            "q3": q3,
            "iqr": float(q3 - q1),
            "n_valid": int(len(vals)),
        })

    return pd.DataFrame(rows)

# ============================================================
# Loaders
# ============================================================
def make_loaders(is_resize: bool):
    dp = DataLoader_Producer(fov=FOV, label_type=LABEL_TYPE, batch_size=BATCH_SIZE, is_resize=is_resize)
    return dp.get_data_loader_ipn_v2()

def select_loader(split, loaders_tuple):
    train_loader, val_loader, test_loader = loaders_tuple
    if split == "train":
        return [("train", train_loader)]
    if split == "val":
        return [("val", val_loader)]
    if split == "test":
        if test_loader is None:
            raise ValueError("test_loader is None")
        return [("test", test_loader)]
    raise ValueError("SPLIT must be train/val/test")

def main():
    if not os.path.isabs(CKPT_DIR) or not os.path.isabs(OUT_ROOT):
        raise ValueError("CKPT_DIR / OUT_ROOT 必须是绝对路径")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"eval_{MODEL_NAME}_{LABEL_TYPE}_{SPLIT}_{run_id}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pattern = os.path.join(CKPT_DIR, "**", PATTERN) if RECURSIVE else os.path.join(CKPT_DIR, PATTERN)
    ckpts = sorted(glob.glob(pattern, recursive=RECURSIVE))
    ckpts = [os.path.abspath(p) for p in ckpts if os.path.isfile(p)]
    if len(ckpts) == 0:
        raise FileNotFoundError(f"在 {CKPT_DIR} 下未找到 {PATTERN}")

    is_resize = not (("SwinSnake" in MODEL_NAME) or ("DSCNet" in MODEL_NAME))
    loaders_tuple = make_loaders(is_resize=is_resize)
    split_loaders = select_loader(SPLIT, loaders_tuple)

    model = build_model(MODEL_NAME, device, HP)
    model.eval()

    all_summary = []
    failed = []

    first_ckpt = ckpts[0]

    for ckpt in ckpts:
        try:
            sd = torch.load(ckpt, map_location="cpu")
            if not isinstance(sd, dict):
                raise ValueError("checkpoint 不是 state_dict(dict)")
            sd = strip_module_prefix(sd)

            ratio = load_state_shape_matched(model, sd)
            if ratio < MIN_MATCH_RATIO:
                raise ValueError(f"load_match_ratio={ratio:.4f} < {MIN_MATCH_RATIO} (结构不一致？)")

            do_vis = not (SKIP_VIS_FOR_FIRST_CKPT and ckpt == first_ckpt)

            ckpt_stem = os.path.splitext(os.path.basename(ckpt))[0]
            vis_dir_ckpt = os.path.join(out_dir, "pred_only", ckpt_stem)

            for split_name, loader in split_loaders:
                recs = []
                pbar = tqdm(loader, desc=f"{os.path.basename(ckpt)} | {split_name}")
                with torch.no_grad():
                    for images, labels, sample_ids in pbar:
                        images = images.to(torch.float).to(device)
                        labels = labels.to(torch.float).to(device)

                        out = model(images)
                        prob = ensure_prob(out)

                        prob_np = to_numpy(prob)
                        gt_np = to_numpy(labels)

                        B = prob_np.shape[0]
                        if isinstance(sample_ids, (list, tuple)):
                            sid_list = [str(x) for x in sample_ids]
                        else:
                            try:
                                sid_list = [str(sample_ids[i]) for i in range(B)]
                            except Exception:
                                sid_list = [str(sample_ids)] * B

                        for i in range(B):
                            p = prob_np[i, 0]
                            g = gt_np[i, 0]

                            pb = binarize(p, THR)
                            gb = binarize(g, 0.5)

                            r = {
                                "ckpt_file": os.path.basename(ckpt),
                                "ckpt_path": ckpt,
                                "model_name": MODEL_NAME,
                                "split": split_name,
                                "sample_id": sid_list[i],
                                "load_match_ratio": ratio,
                            }

                            # ===== 所有指标统一用 pb（已去掉50像素过滤）=====
                            r["Dice"] = dice(pb, gb)
                            r["Jaccard"] = jaccard(pb, gb)
                            r["Precision"] = precision(pb, gb)
                            r["Recall"] = recall(pb, gb)
                            r["Specificity"] = specificity(pb, gb)
                            r["BACC"] = bacc(pb, gb)
                            r["GMean"] = gmean(pb, gb)
                            r["AUC"] = auc_roc(p, gb)

                            r["Hausdorff"] = hausdorff(pb, gb)
                            r["HD95"] = hd95(pb, gb)
                            r["ASSD"] = assd(pb, gb)
                            r["SurfaceDice"] = surface_dice(pb, gb, tol=SURFACE_DICE_TOL)

                            r["Connectivity"] = connectivity_lcc_ratio(pb)
                            r["ComponentCount"] = component_count(pb)
                            r["Betti0Error"] = betti0_error(pb, gb)
                            r["Betti1Error"] = betti1_error(pb, gb)

                            r["clDice"] = cldice_correct(pb, gb)

                            recs.append(r)

                            # ===== 保存预测与可视化（不拼接，仅预测相关）=====
                            if do_vis:
                                sid = _safe_name(sid_list[i])
                                save_dir = os.path.join(vis_dir_ckpt, split_name)
                                base_name = f"{LABEL_TYPE}_pred_{sid}"

                                img_u8 = tensor_image_to_u8_3ch(images[i])
                                save_pred_artifacts(pb, img_u8, save_dir, base_name)

                df = pd.DataFrame(recs)

                if SAVE_PER_SAMPLE:
                    df.to_csv(
                        os.path.join(out_dir, f"{os.path.splitext(os.path.basename(ckpt))[0]}__{split_name}__per_sample.csv"),
                        index=False
                    )

                summ = summarize(df)
                summ.insert(0, "ckpt_file", os.path.basename(ckpt))
                summ.insert(1, "split", split_name)
                summ.insert(2, "ckpt_path", ckpt)
                summ["load_match_ratio"] = ratio
                all_summary.append(summ)

        except Exception as e:
            failed.append({"ckpt_path": ckpt, "ckpt_file": os.path.basename(ckpt), "reason": str(e)})

    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    summary_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)

    failed_df = pd.DataFrame(failed)
    failed_df.to_csv(os.path.join(out_dir, "failed_checkpoints.csv"), index=False)

    if not summary_df.empty:
        pivot_mean = summary_df.pivot(index="ckpt_file", columns="metric", values="mean")
        pivot_std = summary_df.pivot(index="ckpt_file", columns="metric", values="std")

        pivot_median = summary_df.pivot(index="ckpt_file", columns="metric", values="median")
        pivot_q1 = summary_df.pivot(index="ckpt_file", columns="metric", values="q1")
        pivot_q3 = summary_df.pivot(index="ckpt_file", columns="metric", values="q3")

        pivot_mean.to_csv(os.path.join(out_dir, f"comparison_{SPLIT}_pivot_mean.csv"))
        pivot_std.to_csv(os.path.join(out_dir, f"comparison_{SPLIT}_pivot_std.csv"))
        pivot_median.to_csv(os.path.join(out_dir, f"comparison_{SPLIT}_pivot_median.csv"))
        pivot_q1.to_csv(os.path.join(out_dir, f"comparison_{SPLIT}_pivot_q1.csv"))
        pivot_q3.to_csv(os.path.join(out_dir, f"comparison_{SPLIT}_pivot_q3.csv"))

        report = pivot_mean.copy()
        for c in report.columns:
            if c == "HD95":
                med = pivot_median.get(c)
                q1 = pivot_q1.get(c)
                q3 = pivot_q3.get(c)
                report[c] = [
                    f"{mv:.6f} ({a:.6f}, {b:.6f})"
                    if (med is not None and q1 is not None and q3 is not None
                        and not np.isnan(mv) and not np.isnan(a) and not np.isnan(b))
                    else ""
                    for mv, a, b in zip(med, q1, q3)
                ]
            else:
                m = pivot_mean[c]
                s = pivot_std[c]
                report[c] = [
                    f"{mv:.6f}±{sv:.6f}" if (not np.isnan(mv) and not np.isnan(sv)) else ""
                    for mv, sv in zip(m, s)
                ]

        report.to_csv(os.path.join(out_dir, f"comparison_{SPLIT}_pivot_report.csv"))

    print("\n=== DONE ===")
    print("Output folder:", out_dir)
    print("- comparison_summary.csv")
    print(f"- comparison_{SPLIT}_pivot_report.csv")
    print("- failed_checkpoints.csv")

if __name__ == "__main__":
    main()
