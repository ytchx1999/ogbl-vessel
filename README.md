
# SAGE+JKNet in ogbl-vessel

This repo is the code of [SAGE+JKNet](https://arxiv.org/pdf/2006.10637.pdf) model on [ogbl-vessel](https://ogb.stanford.edu/docs/linkprop/#ogbl-vessel) dataset.

Performance on **ogbl-vessel** (10 runs):

| Methods   |  Test Acc  | Valid Acc  |
|  :----  | ---- | ---- |
| SAGE+JKNet |   ±  |  ±  |

<!-- `TGN-no-mem` achieves top-2 performance on DGraphFin until August, 2022. ([DGraph-Fin Leaderboard](https://dgraph.xinye.com/leaderboards/dgraphfin)) -->


## 1. Setup 

### 1.1 Environment

- Dependencies: 
```{bash}
python==3.8
torch==1.10.1+cu102
torch-geometric==2.0.4
```

- GPU: Tesla V100 (32GB)

- Params: 273

### 1.2 Dataset

The dataset [ogbl-vessel](https://ogb.stanford.edu/docs/linkprop/#ogbl-vessel) can be download and placed in `./dataset/ogbl_vessel/`.

## 2. Usage

Full batch GraphSAGE that aggregates the outputs of each layer with JKNet. 
- add JKNet (max).
- reduce the dimension of hidden channels.
- set seed for implement.
- attempt leanrnable embeddings to replace `data.x`, but the performance is low.

We can run `SAGE+JKNet` in 10 runs with seed 0-9: 

```bash
cd scripts/
bash train_sage.sh
```

## 3. Note
The implemention is based on [https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/vessel](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/vessel).