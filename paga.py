# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 17:08
# @Author  : pxh
# @VERSON:1.0
# @File    : paga.py
# @Description : 
import datetime

now1 = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("时间1:", now1)
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

data_root = '/data/pxh/SEDR/data/DLPFC'

proj_list = ['151507']

for proj_idx in range(len(proj_list)):
    data_name = proj_list[proj_idx]
    print('===== Project ' + str(proj_idx + 1) + ' : ' + data_name)
    method = 'louvain'
    parameter = "6louvain,n_epochs=300reg_ssc=0.1,cost_ssc = 0.1"
    adata = sc.read(f"./h5ad2/DLPFC/{data_name}.h5ad")

    if data_name in ['151669', '151670', '151671', '151672']:
        n_clusters = 5
    else:
        n_clusters = 7

    used_adata = adata[adata.obs['ground_truth'] != 'nan',]
    sc.tl.paga(used_adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4, 3)
    sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
                       title=data_name + '_MSGATE',
                       legend_fontoutline=2, show=False)
    plt.savefig(f'./outputs/DLPFC/paga/test.jpg',
                bbox_inches='tight', dpi=300)
    plt.savefig(f'./outputs/DLPFC/paga/{data_name}paga.jpg',
                bbox_inches='tight', dpi=300)
