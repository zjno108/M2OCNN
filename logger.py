import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def create_recordfile(metrics, log_dir="log", prefix=""):
    # 检查log_dir是否存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    recordfile = (
        prefix + "start@" + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + ".csv"
    )
    recordfile = os.path.join(log_dir, recordfile)
    headline = [""]
    headline.extend(metrics)
    headline.extend(["完成时间"])
    with open(recordfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headline)

    return recordfile


def metric_print(res_metric, prefix=""):
    print_line = prefix + "\t"
    for res in res_metric:
        print_line += str(np.round(res, 3)) + "\t"

    print(print_line)


def metric_record(res_metric, recfile, prefix=""):
    rec_line = [prefix]
    for res in res_metric:
        rec_line.append(np.round(res, 5))

    rec_line.append(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    with open(recfile, "a+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(rec_line)


def draw_img(img_group, show_titles, index_list, figsize=(12, 4)):
    nrows = len(index_list)
    ncols = len(img_group)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    for col, title in enumerate(show_titles):
        axs[0, col].set_title(title)

    for row, ind in enumerate(index_list):
        for col, imgs in enumerate(img_group):
            axs[row, col].imshow(imgs[ind], cmap="gray")
            axs[row, col].axis("off")

    fig.tight_layout()

    plt.show()
    plt.close()
