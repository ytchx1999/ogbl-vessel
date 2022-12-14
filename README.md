
# OGBL-Vessel Submission

This repo is the code on [ogbl-vessel](https://ogb.stanford.edu/docs/linkprop/#ogbl-vessel) dataset.

Performance on **ogbl-vessel** (10 runs):

| Methods   |  Test AUC  | Valid AUC  |  Params |
|  :----  | :--: | :--: | :----: |
| SAGE+JKNet (2-layers) |  0.5001 ± 0.0033  | 0.5013 ± 0.0035  | 273 |
| SAGE+JKNet (3-layers) |  0.5001 ± 0.0007 |  0.5014 ± 0.0004  | 481 |
| SAGE+JKNet (4-layers) |  0.5003 ± 0.0005 |  0.5009 ± 0.0003  | 689 |
| **SEAL (no-xfeat)** |  **0.8073 ± 0.0001** |  **0.8077 ± 0.0001**  | 43714 | 

<!-- 86594 -->

<!-- `TGN-no-mem` achieves top-2 performance on DGraphFin until August, 2022. ([DGraph-Fin Leaderboard](https://dgraph.xinye.com/leaderboards/dgraphfin)) -->


## 1. Setup 

### 1.1 Environment

- Dependencies: 
```{bash}
python==3.8
torch==1.10.1+cu102
torch-geometric==2.0.4
ogb==1.3.4
tqdm
```
- GPU: Tesla V100 (32GB)

### 1.2 Dataset

The dataset [ogbl-vessel](https://ogb.stanford.edu/docs/linkprop/#ogbl-vessel) can be download and placed in `./dataset/ogbl_vessel/`.

## 2. Usage

### 2.1 SAGE+JKNet

Full batch [GraphSAGE](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) that aggregates the outputs of each layer with [JKNet](http://proceedings.mlr.press/v80/xu18c/xu18c.pdf). 
- add JKNet (max): 3-layers and 4-layers make the performance more stable (std is lower) compared to raw GraphSAGE.
- reduce the dimension of hidden channels.
- set seed for implement.
- attempt leanrnable embeddings to replace `data.x`, but the performance is low.
<!-- - modify some other hyper-parameters, such as lr. -->

To run `SAGE+JKNet` in 10 runs with seed 0-9: 

```bash
cd scripts/
# 3 layers
bash train_sage.sh
# 4 layers
bash train_sage_4_hop.sh
```

### 2.2 SEAL

To run [SEAL](https://arxiv.org/pdf/2010.16103.pdf) without using `data.x` in 10 runs with seed 0-9: 

```bash
cd scripts/
bash train_seal.sh
```

## 3. Note
The implemention is based on [https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/vessel](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/vessel) and [https://github.com/facebookresearch/SEAL_OGB](https://github.com/facebookresearch/SEAL_OGB).