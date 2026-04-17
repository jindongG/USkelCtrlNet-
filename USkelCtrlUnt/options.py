import argparse

parser = argparse.ArgumentParser(description='training argument values')

# training:
parser.add_argument("-device", type=str, default="0")
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=1)
parser.add_argument("-lr", type=float, default=0.0001)
parser.add_argument("-k_fold", type=int, default=10)
parser.add_argument("-save_weight", type=bool, default=True)

# dataset:
parser.add_argument("-fov", type=str, default="6M")
parser.add_argument("-label_type", type=str, default="LargeVessel") # "LargeVessel", "FAZ", "Artery", "Vein", "Capillary"

# deep model:
parser.add_argument("-model_name", type=str, default="SwinSnake_Alter")#OCTAMamba,SwinSnake_Dual,DSCNet,DiNTS,SegResNet,UNet,FlexUNet,SwinUNETR,SwinSnake_Alter
parser.add_argument("-layer_depth", type=int, default=2)
parser.add_argument("-kernel_size", type=int, default=11)
parser.add_argument("-extend_scope", type=int, default=1)
parser.add_argument("-down_layer", type=str, default="MaxPooling")#DeformDown  MaxPooling  DirectionalPooling
parser.add_argument("-rate", type=int, default=72)
parser.add_argument("-repeat_n", type=int, default=1)

parser.add_argument('--alpha', type=float, default=0.6)
parser.add_argument('--beta', type=float, default=0.25)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--delta', type=float, default=0.05)
parser.add_argument('--ohem_weight', type=float, default=0.5)
parser.add_argument('--ohem_top_percent', type=float, default=0.25)#0.25

parser.add_argument("--eval_thr", type=float, default=0.5,
                    help="Threshold for binarizing pred_prob during evaluation (probability space).")

# evaluation
parser.add_argument(
    "-metrics", nargs="+",
    default=[
        "Dice", "Jaccard",
        "Precision", "Recall", "Specificity",
        "Hausdorff", "HD95",
        "clDice", "AUC",
        "Connectivity", "ComponentCount",
        "Betti0Error", "Betti1Error"
    ]
)

# others:
parser.add_argument("-remark", type=str, default="#")


args = parser.parse_args()
