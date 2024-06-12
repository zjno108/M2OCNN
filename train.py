import csv
import glob
import os
import warnings
from datetime import datetime

import imageio.v2 as imageio
import numpy as np
import torch
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SFModel
from utils.dataset import CreateDataset
from utils.loss import DcmpLoss, Fusionloss, GradLoss, SSIMLoss
from utils.matrics import compute_similarity_metrics_arr
from utils.misc import tensors2imgarr

warnings.filterwarnings("ignore")

now = datetime.now()
time_train = now.strftime("%m-%d-%H-%M")

image_set = "/home/zhangjian/Repositories/Datasets/I2I/BraTS2023"
contrast1 = "t1c"
contrast2 = "t1n"
# image_set = "/home/zhangjian/Repositories/Datasets/I2I/IXI/"
# contrast1 = "t2"
# contrast2 = "pd"

savedir = os.path.join(
    "/home/zhangjian/Repositories/DSLFuse2/log/BraTS2023/",
    contrast1 + "_" + contrast2,
    time_train,
)
# savedir = os.path.join(
#     "/home/zhangjian/Repositories/DSLFuse2/log/IXI/",
#     contrast1 + "_" + contrast2,
#     time_train,
# )

imsavedir = os.path.join(savedir, "imsave")
if not os.path.exists(imsavedir):
    os.makedirs(imsavedir)
modelsave = os.path.join(savedir, "modelsave")
if not os.path.exists(modelsave):
    os.makedirs(modelsave)
resdir = os.path.join(savedir, "result")
if not os.path.exists(resdir):
    os.makedirs(resdir)


dataset = CreateDataset(
    phase="train", input_path=image_set, contrast1=contrast1, contrast2=contrast2
)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

val_dataset = CreateDataset(
    phase="val", input_path=image_set, contrast1=contrast1, contrast2=contrast2
)
val_data = DataLoader(val_dataset, batch_size=1, shuffle=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lr = 1e-4
w_decay = 0.01
step = 10
gamma = 0.1

model = SFModel()
optm = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optm, step_size=step, gamma=gamma)
model = model.to(device)

criteria_mse = nn.MSELoss().cuda()
criteria_l1 = nn.L1Loss().cuda()
criteria_ssim = SSIMLoss().cuda()
criteria_grad = GradLoss().cuda()
criteria_fuse = Fusionloss().cuda()
criteria_dcmp = DcmpLoss().cuda()


num_epochs = 60

ssim_coeff = 0.1
dcmp_coeff = 0.1

for epoch in range(num_epochs):
    model.train()

    loss_epoch = 0

    for a, b in tqdm(data_loader):
        optm.zero_grad()
        ra, rb = a.to(device), b.to(device)

        comm_feat_a, spec_feat_a = model.comm_enc(ra), model.spec_enc(ra)
        comm_feat_b, spec_feat_b = model.comm_enc(rb), model.spec_enc(rb)

        loss_dcmp = criteria_dcmp(comm_feat_a, comm_feat_b, spec_feat_a, spec_feat_b)

        fb = model.dec(torch.cat((comm_feat_a, spec_feat_a), dim=1))
        fa = model.dec(torch.cat((comm_feat_b, spec_feat_b), dim=1))

        loss_a = criteria_l1(ra, fa) + ssim_coeff * criteria_ssim(ra, fa)
        loss_b = criteria_l1(rb, fb) + ssim_coeff * criteria_ssim(rb, fb)
        loss = loss_a + loss_b + dcmp_coeff * loss_dcmp

        ab_all = np.hstack(tensors2imgarr((ra, fa, rb, fb)))
        imageio.imwrite("train_show" + ".png", ab_all)

        loss_epoch += loss.item()

        loss.backward()
        optm.step()

    print("loss =", loss_epoch / len(data_loader))

    with torch.no_grad():
        model.eval()

        metric_a = []
        metric_b = []

        for i, ab in enumerate(tqdm(val_data)):
            a, b = ab
            ra, rb = a.to(device), b.to(device)

            comm_feat_a, spec_feat_a = model.comm_enc(ra), model.spec_enc(ra)
            comm_feat_b, spec_feat_b = model.comm_enc(rb), model.spec_enc(rb)

            fb = model.dec(torch.cat((comm_feat_a, spec_feat_a), dim=1))
            fa = model.dec(torch.cat((comm_feat_b, spec_feat_b), dim=1))

            ra, fa, rb, fb = tensors2imgarr((ra, fa, rb, fb))
            imageio.imwrite(
                os.path.join(savedir, "imsave/ab_" + str(i) + ".png"),
                np.hstack((ra, fa, rb, fb)),
            )
            metric_a.append(compute_similarity_metrics_arr(ra, fa))
            metric_b.append(compute_similarity_metrics_arr(rb, fb))

        metric_a = np.array(metric_a)
        metric_a_mean = np.mean(metric_a, axis=0)
        metric_a_std = np.std(metric_a, axis=0)

        metric_b = np.array(metric_b)
        metric_b_mean = np.mean(metric_b, axis=0)
        metric_b_std = np.std(metric_b, axis=0)

        # 打印 metric
        table = PrettyTable(["psnr", "psnr_nbg", "ssim", "mse", "mae", "nmi", "vif"])
        table.add_row(["a", "", "", "", "", "", ""])
        table.add_row(metric_a_mean)
        table.add_row(metric_a_std)
        table.add_row(["b", "", "", "", "", "", ""])
        table.add_row(metric_b_mean)
        table.add_row(metric_b_std)
        print(table)

        # 保存 metric
        with open(os.path.join(resdir, "a_epoch_" + str(epoch + 1) + ".csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerows(metric_a)
            writer.writerow(metric_a_mean)
            writer.writerow(metric_a_std)

        with open(os.path.join(resdir, "b_epoch_" + str(epoch + 1) + ".csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerows(metric_b)
            writer.writerow(metric_b_mean)
            writer.writerow(metric_b_std)

        print("\n====================", epoch + 1, "====================\n")
    if scheduler is not None:
        scheduler.step()

torch.save(
    model.state_dict(),
    os.path.join(savedir, "modelsave/modelsynt.pth"),
)
