import time

from server.cgmr import CGMR
from options import args_parser
from model.multi_net import Multi_Net
from model.multiunet import MultiUnet

def run(args):
    if args.algorithm == "CGMR":
        if args.mode == 'hsi':
            model = Multi_Net(num_classes=args.num_classes, patch_size=11, encoder_dim=64, depth=2, c1=15, c2=1, c3=3).to(args.device)
        if args.mode == 'rgb':
            model = MultiUnet(class_num=args.num_classes,channel_rgb=3, channel_lidar=1).to(args.device)
        server = CGMR(args,model)
    server.train()

if __name__ == "__main__":
    total_start = time.time()
    args = args_parser()

    run(args)

