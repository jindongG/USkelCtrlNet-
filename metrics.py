import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from collections import defaultdict
from statistics import mean
#from skimage.morphology import skeletonize, label
# 顶部修改导入：把 label 重命名为 sk_label
from skimage.morphology import skeletonize, label as sk_label
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_fill_holes
from collections import defaultdict
from statistics import mean
from skimage.morphology import skeletonize, label as sk_label

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

class MetricsStatistics:
    def __init__(self, save_dir="./results/"):
        self.epsilon = 1e-6
        self.func_dct = {
            "Dice": self.cal_dice,
            "Jaccard": self.cal_jaccard_index,
            "Precision": self.cal_precision,
            "Recall": self.cal_recall,
            "Specificity": self.cal_specificity,

            "BACC": self.cal_bacc,
            "GMean": self.cal_gmean,

            "Hausdorff": self.cal_hausdorff,
            "HD95": self.cal_hd95,
            "ASSD": self.cal_assd,
            "SurfaceDice": self.cal_surface_dice,

            "clDice": self.cal_cldice_correct,
            "AUC": self.cal_auc,

            "Connectivity": self.cal_connectivity,
            "ComponentCount": self.cal_component_count,
            "Betti0Error": self.cal_betti0_error,
            "Betti1Error": self.cal_betti1_error,
        }

        self.save_dir = save_dir
        self.metric_values = defaultdict(list)  # 存储当前epoch的所有样本值
        self.metric_epochs = defaultdict(list)  # 每个epoch的平均值

    # ----------------------------
    # 每epoch保存结果
    # ----------------------------
    #def cal_epoch_metric(self, metrics, label_type, label, pred):
        #for x in metrics:
            #if x not in self.func_dct:
                #continue
            #try:
                #self.metric_values[f"{x}-{label_type}"].append(self.func_dct[x](label, pred))
            #except Exception:
    def cal_dice(self, pred, label):
        intersection = (pred & label).sum().item()
        union = pred.sum().item() + label.sum().item()
        return 2 * intersection / (union + self.epsilon)
            #pass
    def cal_epoch_metric(self, metrics, label_type, label, pred, pred_prob=None):
        """
        label, pred: 二值mask (torch int/bool)
        pred_prob: 概率图 (torch float 0~1)，只给AUC用
        """
        for x in metrics:
            if x not in self.func_dct:
                continue
            try:
                if x == "AUC":
                    if pred_prob is None:
                        continue
                    self.metric_values[f"{x}-{label_type}"].append(self.func_dct[x](pred_prob, label))
                else:
                    self.metric_values[f"{x}-{label_type}"].append(self.func_dct[x](pred, label))
            except Exception as e:
                print(f"[MetricsStatistics] Error computing metric {x} for {label_type}: {e}")
                continue

    def record_result(self, epoch):
        self.metric_epochs["epoch"].append(epoch)
        for k, v in self.metric_values.items():
            if len(v) > 0:
                self.metric_epochs[k].append(mean(v))
        pd.DataFrame(self.metric_epochs).to_excel(f"{self.save_dir}/metrics_statistics.xlsx", index=False)
        self.metric_values.clear()

    # ----------------------------
    # 混淆矩阵类指标
    # ----------------------------
    def cal_confusion_matrix(self, pred, label):
        TP = ((pred == 1) & (label == 1)).sum().item()
        FP = ((pred == 1) & (label == 0)).sum().item()
        FN = ((pred == 0) & (label == 1)).sum().item()
        TN = ((pred == 0) & (label == 0)).sum().item()
        return TP, FP, FN, TN

    def cal_precision(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TP / (TP + FP + self.epsilon)

    def cal_recall(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TP / (TP + FN + self.epsilon)

    def cal_specificity(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        return TN / (TN + FP + self.epsilon)

    # ----------------------------
    # 基础面积指标
    # ----------------------------
    def cal_jaccard_index(self, pred, label):
        intersection = (pred & label).sum().item()
        union = (pred | label).sum().item()
        return intersection / (union + self.epsilon)

    def cal_cldice_correct(self, pred, label):
        p = pred.cpu().numpy().astype(bool)
        g = label.cpu().numpy().astype(bool)

        if p.sum() == 0 and g.sum() == 0:
            return 1.0
        if p.sum() == 0 or g.sum() == 0:
            return 0.0

        sp = skeletonize(p)
        sg = skeletonize(g)
        if sp.sum() == 0 or sg.sum() == 0:
            return 0.0

        tprec = np.logical_and(sp, g).sum() / (sp.sum() + self.epsilon)
        tsens = np.logical_and(sg, p).sum() / (sg.sum() + self.epsilon)
        return (2 * tprec * tsens) / (tprec + tsens + self.epsilon)

    def cal_hausdorff(self, pred, label):
        array1 = pred.cpu().numpy().astype(np.uint8)
        array2 = label.cpu().numpy().astype(np.uint8)
        # 如果两个都为空则返回0
        if array1.sum() == 0 and array2.sum() == 0:
            return 0.0
        try:
            dist1 = directed_hausdorff(np.argwhere(array1), np.argwhere(array2))[0]
            dist2 = directed_hausdorff(np.argwhere(array2), np.argwhere(array1))[0]
            return max(dist1, dist2)
        except Exception:
            return np.nan

    def cal_bacc(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        pos = TP + FN
        neg = TN + FP
        sens = TP / (pos + self.epsilon) if pos > 0 else 1.0
        spec = TN / (neg + self.epsilon) if neg > 0 else 1.0
        return 0.5 * (sens + spec)

    def cal_gmean(self, pred, label):
        TP, FP, FN, TN = self.cal_confusion_matrix(pred, label)
        pos = TP + FN
        neg = TN + FP
        sens = TP / (pos + self.epsilon) if pos > 0 else 1.0
        spec = TN / (neg + self.epsilon) if neg > 0 else 1.0
        return float(np.sqrt(max(sens * spec, 0.0)))

    def cal_assd(self, pred, label):
        p = pred.cpu().numpy().astype(bool)
        g = label.cpu().numpy().astype(bool)
        if p.sum() == 0 and g.sum() == 0:
            return 0.0
        if p.sum() == 0 or g.sum() == 0:
            return np.nan
        bp = self._border(p)
        bg = self._border(g)
        if bp.sum() == 0 or bg.sum() == 0:
            return np.nan
        dt_g = distance_transform_edt(~bg)
        d_p2g = dt_g[bp]
        dt_p = distance_transform_edt(~bp)
        d_g2p = dt_p[bg]
        if d_p2g.size == 0 or d_g2p.size == 0:
            return np.nan
        return float(0.5 * (d_p2g.mean() + d_g2p.mean()))

    def cal_surface_dice(self, pred, label, tol=1.0):
        p = pred.cpu().numpy().astype(bool)
        g = label.cpu().numpy().astype(bool)
        if p.sum() == 0 and g.sum() == 0:
            return 1.0
        if p.sum() == 0 or g.sum() == 0:
            return 0.0
        bp = self._border(p)
        bg = self._border(g)
        if bp.sum() == 0 and bg.sum() == 0:
            return 1.0
        if bp.sum() == 0 or bg.sum() == 0:
            return 0.0
        dt_bg = distance_transform_edt(~bg)
        dt_bp = distance_transform_edt(~bp)
        d_p2g = dt_bg[bp]
        d_g2p = dt_bp[bg]
        if d_p2g.size == 0 or d_g2p.size == 0:
            return 0.0
        m1 = (d_p2g <= tol).sum()
        m2 = (d_g2p <= tol).sum()
        return float((m1 + m2) / (bp.sum() + bg.sum() + self.epsilon))

    # ----------------------------
    # 新增：拓扑结构指标
    # ----------------------------
    def cal_cldice(self, pred, label):
        """骨架一致性 Dice (clDice-like 指标)"""
        pred_np = pred.cpu().numpy().astype(bool)
        label_np = label.cpu().numpy().astype(bool)
        if pred_np.sum() == 0 and label_np.sum() == 0:
            return 1.0
        pred_skel = skeletonize(pred_np)
        label_skel = skeletonize(label_np)
        intersection = np.logical_and(pred_skel, label_skel).sum()
        denom = pred_skel.sum() + label_skel.sum() + self.epsilon
        return 2 * intersection / denom

    def cal_connectivity(self, pred, label):
        """最大连通分量占比"""
        pred_np = pred.cpu().numpy().astype(np.uint8)
        if pred_np.sum() == 0:
            return 0.0
        # 使用 sk_label，避免与变量名冲突
        labeled = sk_label(pred_np, connectivity=2)
        counts = np.bincount(labeled.flatten())[1:]  # 跳过背景
        if len(counts) == 0:
            return 0.0
        largest = counts.max()
        return largest / (pred_np.sum() + self.epsilon)

    def cal_component_count(self, pred, label):
        """连通组件数量"""
        pred_np = pred.cpu().numpy().astype(np.uint8)
        labeled = sk_label(pred_np, connectivity=2)
        return int(labeled.max())

    def _border(self, mask_bool: np.ndarray):
        er = binary_erosion(mask_bool)
        return np.logical_xor(mask_bool, er)

    def cal_hd95(self, pred, label):
        p = pred.cpu().numpy().astype(bool)
        g = label.cpu().numpy().astype(bool)

        if p.sum() == 0 and g.sum() == 0:
            return 0.0
        if p.sum() == 0 or g.sum() == 0:
            return np.nan

        bp = self._border(p)
        bg = self._border(g)
        if bp.sum() == 0 or bg.sum() == 0:
            return np.nan

        dt_g = distance_transform_edt(~g)
        d_p2g = dt_g[bp]
        dt_p = distance_transform_edt(~p)
        d_g2p = dt_p[bg]

        if len(d_p2g) == 0 or len(d_g2p) == 0:
            return np.nan

        return float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))

    def cal_auc(self, pred_prob, label):
        if roc_auc_score is None:
            return np.nan
        y_true = label.cpu().numpy().astype(np.uint8).reshape(-1)
        y_score = pred_prob.cpu().numpy().astype(np.float32).reshape(-1)
        if np.unique(y_true).size < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))

    def _count_holes(self, mask_bool: np.ndarray):
        filled = binary_fill_holes(mask_bool)
        holes = np.logical_and(filled, ~mask_bool)
        lab = sk_label(holes.astype(np.uint8), connectivity=2)
        return int(lab.max())

    def cal_betti0_error(self, pred, label):
        p_cc = int(sk_label(pred.cpu().numpy().astype(np.uint8), connectivity=2).max())
        g_cc = int(sk_label(label.cpu().numpy().astype(np.uint8), connectivity=2).max())
        return abs(p_cc - g_cc)

    def cal_betti1_error(self, pred, label):
        p = pred.cpu().numpy().astype(bool)
        g = label.cpu().numpy().astype(bool)
        return abs(self._count_holes(p) - self._count_holes(g))
