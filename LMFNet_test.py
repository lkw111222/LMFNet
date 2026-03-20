# Enhanced MCFNet_test.py with performance metrics, FLOPs, Params, and PR curve

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
import time
import matplotlib.pyplot as plt
import csv
from thop import profile
from torch.nn import Upsample
from models.LMFNet import LMFNet
from data import test_dataset
from utils.metrics import cal_mae, cal_fm, cal_sm, cal_em, cal_pr

# Custom hook to skip FLOPs counting for Upsample
def count_upsample(m, x, y):
    m.total_ops += torch.zeros(1).to(y.device)

custom_ops = {Upsample: count_upsample}

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./test/',help='test dataset path')
opt = parser.parse_args()

if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

model = LMFNet()
model.load_state_dict(torch.load('./cpts/MyNet_epoch_best.pth'))
model.eval()

# Step 1: 用 CPU 模型和 CPU 输入统计 FLOPs & Params
model_cpu = LMFNet()
model_cpu.load_state_dict(torch.load('./cpts/MyNet_epoch_best.pth', map_location='cpu'))
model_cpu.eval()

input_rgb = torch.randn(1, 3, opt.testsize, opt.testsize)  # 不需要 .cuda()
input_d = torch.randn(1, 3, opt.testsize, opt.testsize)

flops, params = profile(model_cpu, inputs=(input_rgb, input_d), custom_ops=custom_ops)

print(f"\nModel Complexity:\nParams: {params / 1e6:.2f}M\nFLOPs: {flops / 1e9:.2f}G\n")

# Step 2: 真正运行时再用 GPU 模型推理
model = LMFNet
model.load_state_dict(torch.load('./cpts/MyNet_epoch_best.pth'))
model.cuda()
model.eval()


print(f"\nModel Complexity:\nParams: {params / 1e6:.2f}M\nFLOPs: {flops / 1e9:.2f}G\n")

# Dataset loop
test_datasets = ['LFSD','DUT-RGBD','NLPR','SIP','STERE','NJU2K']
all_results = []

for dataset in test_datasets:
    print(f"\nTesting on {dataset}...")
    image_root = os.path.join(opt.test_path, dataset, 'RGB/')
    gt_root = os.path.join(opt.test_path, dataset, 'GT/')
    depth_root = os.path.join(opt.test_path, dataset, 'depth/')
    save_path = './pre_maps/' + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

    mae_sum = 0
    sm_sum = 0
    em_sum = 0
    fm_sum = 0
    precision_all = []
    recall_all = []
    cost_time = []

    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1,3,1,1).cuda()

        start_time = time.time()
        res,_,_,_ = model(image, depth)
        cost_time.append(time.time() - start_time)

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        cv2.imwrite(save_path + name, res * 255)

        mae_sum += cal_mae(res, gt)
        sm_sum += cal_sm(res, gt)
        em_sum += cal_em(res, gt)
        fm_sum += cal_fm(res, gt)
        p, r = cal_pr(res, gt)
        precision_all.append(p)
        recall_all.append(r)

    mean_mae = mae_sum / test_loader.size
    mean_sm = sm_sum / test_loader.size
    mean_em = em_sum / test_loader.size
    mean_fm = fm_sum / test_loader.size
    mean_fps = test_loader.size / np.sum(cost_time)

    precision = np.mean(precision_all, axis=0)
    recall = np.mean(recall_all, axis=0)

    # Draw PR curve
    plt.figure()
    plt.plot(recall, precision, label=dataset)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {dataset}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'PR_Curve_{dataset}.png')
    plt.close()

    print(f"Dataset: {dataset}\nMAE: {mean_mae:.4f}, S-measure: {mean_sm:.4f}, E-measure: {mean_em:.4f}, F-measure: {mean_fm:.4f}, FPS: {mean_fps:.2f}")

    all_results.append([dataset, mean_mae, mean_sm, mean_em, mean_fm, mean_fps, params / 1e6, flops / 1e9])

# Save to CSV
with open('test_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'MAE', 'S-measure', 'E-measure', 'F-measure', 'FPS', 'Params(M)', 'FLOPs(G)'])
    writer.writerows(all_results)

print("\nAll results saved to test_results.csv")