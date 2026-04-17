import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import L1Loss
import os, time, random, numpy as np
from tqdm import tqdm
from options import *
from loss_functions import *
from dataset import DataLoader_Producer
from metrics import MetricsStatistics
from monai.networks.nets import *
#from models.SwinSnake import SwinSnake_Alter  # , SwinSnake_Dual
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
import pandas as pd
#from models.octamamba import OCTAMamba
from models.SwinSnake1 import DSCNet, SwinSnake_Alter, SwinSnake_Dual
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        return self.new_layer(self.original_model(x))


class TrainManager:
    def __init__(self, model_dct, dataloader_producer):
        time_str = "-".join(["{:0>2}".format(x) for x in time.localtime(time.time())][:-3])
        config_str = "_".join(map(str, [*model_dct.values(), args.fov, args.label_type, args.epochs, args.remark]))
        self.record_dir = os.path.join("results", time_str, config_str)
        self.cpt_dir = os.path.join(self.record_dir, "checkpoints")
        os.makedirs(self.cpt_dir, exist_ok=True)
        self.model = self.get_model(model_dct)
        self.dataloader_producer = dataloader_producer
        self.save_weight = model_dct["save_weight"]
        if self.save_weight:
            torch.save(self.model.state_dict(), f"{self.cpt_dir}/init.pth")

        # 初始损失模块
        self.total_loss_module = TotalVesselLoss(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta
        )
        self.ohem_weight = args.ohem_weight
        self.ohem_top_percent = args.ohem_top_percent
        self.inputs_process = lambda x: x.to(torch.float).to(device)

    def full_metrics(self):
        return [
            "Dice", "Jaccard",
            "Precision", "Recall", "Specificity",
            "BACC", "GMean",
            "Hausdorff", "HD95",
            "ASSD", "SurfaceDice",
            "clDice", "AUC",
            "Connectivity", "ComponentCount",
            "Betti0Error", "Betti1Error",
        ]

    def summarize_epoch_samples(self, df: pd.DataFrame, save_dir: str, prefix: str):
        """
        对齐 eval 脚本 summarize() 的统计逻辑
        """
        METRICS = [c for c in df.columns if c not in ["sample_id", "loader"]]

        rows = []
        for m in METRICS:
            vals = df[m].to_numpy()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                rows.append(dict(metric=m, mean=np.nan, std=np.nan,
                                 median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan, n_valid=0))
                continue
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            rows.append(dict(
                metric=m,
                mean=float(np.mean(vals)),
                std=float(np.std(vals)),
                median=float(np.median(vals)),
                q1=q1, q3=q3,
                iqr=float(q3 - q1),
                n_valid=int(len(vals)),
            ))

        summ = pd.DataFrame(rows)
        summ.to_csv(os.path.join(save_dir, f"{prefix}_summary.csv"), index=False)
        return summ

    # -----------------------------
    # 模型选择
    # -----------------------------
    def get_model(self, model_dct):
        name = model_dct["name"]
        if "SwinSnake_Alter" == model_dct["name"]:
            model = SwinSnake_Alter(img_ch=3, output_ch=1, layer_depth=model_dct["layer_depth"],
                                    kernel_size=model_dct["kernel_size"], extend_scope=model_dct["extend_scope"],
                                    down_layer=model_dct["down_layer"],
                                    rate=model_dct["rate"], repeat_n=model_dct["repeat_n"], device_id=args.device).to(
                device)
        elif model_dct["name"] == "FlexUNet":
            model = FlexUNet(in_channels=3, out_channels=1, spatial_dims=2, backbone="efficientnet-b4")
            model = ModifiedModel(model).to(device)
        elif "SwinSnake_Dual" == model_dct["name"]:
            model = SwinSnake_Dual(img_ch=3, output_ch=1, layer_depth=model_dct["layer_depth"],
                                   kernel_size=model_dct["kernel_size"], extend_scope=model_dct["extend_scope"],
                                   rate=model_dct["rate"], device_id=args.device).to(device)

        elif "DSCNet" == model_dct["name"]:
            model = DSCNet(img_ch=3, output_ch=1, layer_depth=model_dct["layer_depth"],
                           kernel_size=model_dct["kernel_size"], extend_scope=model_dct["extend_scope"],
                           rate=model_dct["rate"], device_id=args.device).to(device)

        elif model_dct["name"] == "SwinUNETR":
            model = SwinUNETR(in_channels=3, out_channels=1, feature_size=24 * model_dct["layer_depth"], spatial_dims=2)
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "SegResNet":
            model = SegResNet(in_channels=3, out_channels=1, spatial_dims=2)
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "FlexUNet":
            model = FlexUNet(in_channels=3, out_channels=1, spatial_dims=2, backbone="efficientnet-b4")
            model = ModifiedModel(model).to(device)

        elif model_dct["name"] == "DiNTS":
            dints_space = TopologyInstance(spatial_dims=2, device="cuda:{}".format(args.device))
            model = DiNTS(dints_space=dints_space, in_channels=3, num_classes=1, spatial_dims=2)
            model = ModifiedModel(model).to(device)
        elif name == "UNet":
            N, B = 5, 8
            channels = [2 ** x for x in range(B, B + N)]
            strides = [2] * (len(channels) - 1)
            model = ModifiedModel(
                UNet(
                    in_channels=3,
                    out_channels=1,
                    spatial_dims=2,
                    channels=channels,
                    strides=strides
                )
            ).to(device)

        else:
            raise ValueError(f"Unsupported model type: {name}")
        return model

    # -----------------------------
    # 优化器 + warmup + cosine LR
    # -----------------------------
    def reset(self):
        if self.save_weight:
            self.model.load_state_dict(torch.load(f"{self.cpt_dir}/init.pth"))
        pg = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )

    # -----------------------------
    # 确保预测为概率
    # -----------------------------
    def _ensure_prob(self, preds):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        return preds.clamp(0.0, 1.0) if preds.max() <= 1.0 else torch.sigmoid(preds)

    # -----------------------------
    # 训练主循环
    # -----------------------------
    def train(self):
        train_loader, val_loader, test_loader = self.dataloader_producer.get_data_loader_ipn_v2()
        self.reset()
        metrics_statistics = MetricsStatistics(save_dir=self.record_dir)
        self.record_performance(train_loader, val_loader, test_loader, 0, metrics_statistics)
        warmup_epochs = int(args.epochs * 0.1)  # 前10% warm-up

        for epoch in tqdm(range(1, args.epochs + 1), desc="training"):
            self.model.train()
            total_epoch_loss = 0

            # 动态权重策略
            if epoch <= warmup_epochs:
                self.total_loss_module.alpha, self.total_loss_module.beta = 0.7, 0.3
                self.total_loss_module.gamma, self.total_loss_module.delta = 0.05, 0
                ohem_weight = 0.0
            elif epoch <= args.epochs * 0.5:
                self.total_loss_module.alpha, self.total_loss_module.beta = 0.6, 0.3
                self.total_loss_module.gamma, self.total_loss_module.delta = 0.1, 0
                ohem_weight = 0.5
            else:
                self.total_loss_module.alpha, self.total_loss_module.beta = 0.5, 0.4
                self.total_loss_module.gamma, self.total_loss_module.delta = 0.1, 0
                ohem_weight = 0.7

            for samples, labels, _ in train_loader:
                samples, labels = map(self.inputs_process, (samples, labels))
                self.optimizer.zero_grad()
                preds = self.model(samples)
                pred_prob = self._ensure_prob(preds)
                main_loss_val, _ = self.total_loss_module(pred_prob, labels)
                ohem_loss = pixel_ohem_loss_from_prob(pred_prob, labels, top_percent=self.ohem_top_percent)
                total_loss = main_loss_val + ohem_weight * ohem_loss
                total_loss.backward()
                self.optimizer.step()
                total_epoch_loss += total_loss.item()

            # Warm-up 手动线性调升 LR
            if epoch <= warmup_epochs:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = args.lr * (epoch / warmup_epochs)
            else:
                self.scheduler.step()

            avg_train_loss = total_epoch_loss / len(train_loader)
            print(f"\n[Epoch {epoch}/{args.epochs}] Train Loss: {avg_train_loss:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.6e}")

            # ✅ 每 5 个 epoch 记录一次验证指标
            if epoch % 5 == 0 or epoch == args.epochs:
                val_metrics = self.record_performance(train_loader, val_loader, test_loader, epoch, metrics_statistics)
                print(" Validation Metrics:")
                for k, v in val_metrics.items():
                    if any(m in k for m in ["Dice", "clDice", "BACC", "GMean", "loss_val"]):
                        print(f" {k:<35}: {v:.4f}")

    # -----------------------------
    # 验证与保存
    # -----------------------------
    def record_performance(self, train_loader, val_loader, test_loader, epoch, metrics_statistics):
        save_dir = f"{self.record_dir}/{epoch:0>4}"
        # ===== 新增：是否保存分割 npy =====
        SAVE_NPY_EPOCH = (epoch % 5 == 0 or epoch == args.epochs)

        if self.save_weight:
            torch.save(self.model.state_dict(), f"{self.cpt_dir}/{epoch:0>4}.pth")

        metrics_statistics.metric_values["learning rate"].append(
            self.optimizer.param_groups[0]['lr']
        )

        epoch_metrics_dict = {}
        per_sample_records = []

        def metrics_for(loader_type: str):
            if loader_type == "train":
                return [
                    "Dice", "clDice", "BACC", "GMean"
                ]

            if loader_type == "val":
                return [
                    "Dice",
                    "BACC",  "GMean",
                    "clDice",
                ]

            if loader_type == "test":
                return self.full_metrics()

        def record_dataloader(dataloader, loader_type):
            if dataloader is None:
                return

            with torch.no_grad():
                for images, labels, sample_ids in dataloader:
                    images, labels = map(self.inputs_process, (images, labels))

                    out = self.model(images)
                    preds = out[0] if isinstance(out, (tuple, list)) else out
                    pred_prob = self._ensure_prob(preds)

                    pred_bin = (pred_prob > args.eval_thr).int()

                    # ✅ 对齐训练端 GT 口径
                    label_bin = labels.int()

                    pred_1 = pred_bin[0, 0].cpu()
                    lab_1 = label_bin[0, 0].cpu()
                    prob_1 = pred_prob[0, 0].cpu()
                    # ===== 新增：保存分割结果 npy（仅 test + 每5个epoch）=====
                    if loader_type == "test" and SAVE_NPY_EPOCH:
                        os.makedirs(save_dir, exist_ok=True)

                        sid = str(sample_ids[0])

                        # ---------- image ----------
                        # images: [1, 3, H, W] -> [3, H, W], uint8
                        img_np = images[0].detach().cpu().numpy()
                        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                        # ---------- label ----------
                        # [H, W], 0/255
                        label_np = (lab_1.numpy().astype(np.uint8) * 255)

                        # ---------- pred ----------
                        # [H, W], 0/255
                        pred_np = (pred_1.numpy().astype(np.uint8) * 255)

                        np.save(f"{save_dir}/{args.label_type}_sample_{sid}.npy", img_np)
                        np.save(f"{save_dir}/{args.label_type}_label_{sid}.npy", label_np)
                        np.save(f"{save_dir}/{args.label_type}_pred_{sid}.npy", pred_np)

                    metrics_statistics.cal_epoch_metric(
                        metrics_for(loader_type),
                        f"{args.label_type}-{loader_type}",
                        lab_1,
                        pred_1,
                        pred_prob=prob_1
                    )

                    # ✅ 只在 test 记录 per-sample
                    if loader_type == "test":
                        sample_id = str(sample_ids[0])
                        record = {
                            "epoch": epoch,
                            "loader": loader_type,
                            "sample_id": sample_id,
                        }

                        for m in self.full_metrics():
                            try:
                                if m == "AUC":
                                    record[m] = metrics_statistics.func_dct[m](prob_1, lab_1)
                                else:
                                    record[m] = metrics_statistics.func_dct[m](pred_1, lab_1)
                            except Exception:
                                record[m] = np.nan

                        per_sample_records.append(record)

        record_dataloader(train_loader, "train")
        record_dataloader(val_loader, "val")
        record_dataloader(test_loader, "test")

        metrics_statistics.record_result(epoch)

        # ===== test per-sample CSV & summary =====
        if len(per_sample_records) > 0:
            os.makedirs(save_dir, exist_ok=True)
            df_samples = pd.DataFrame(per_sample_records)

            df_samples.to_csv(
                os.path.join(save_dir, f"{args.label_type}_epoch_{epoch:04d}_test_per_sample.csv"),
                index=False
            )

            self.summarize_epoch_samples(
                df_samples,
                save_dir,
                prefix=f"{args.label_type}_epoch_{epoch:04d}_test"
            )

        # 返回 epoch mean 指标
        if len(metrics_statistics.metric_epochs["epoch"]) > 0:
            for k, v in metrics_statistics.metric_epochs.items():
                if k != "epoch":
                    epoch_metrics_dict[k] = v[-1] if len(v) > 0 else np.nan

        return epoch_metrics_dict


if __name__ == "__main__":
    model_dct = {
        "name": args.model_name,
        "layer_depth": args.layer_depth,
        "kernel_size": args.kernel_size,
        "extend_scope": args.extend_scope,
        "rate": args.rate,
        "down_layer": args.down_layer,
        "repeat_n": args.repeat_n,
        "save_weight": args.save_weight
    }
    is_resize = not ("SwinSnake" in model_dct["name"] or "DSCNet" in model_dct["name"])
    dataloader_producer = DataLoader_Producer(
        fov=args.fov,
        label_type=args.label_type,
        batch_size=args.batch_size,
        is_resize=is_resize
    )
    train_manager = TrainManager(model_dct=model_dct, dataloader_producer=dataloader_producer)
    train_manager.train()