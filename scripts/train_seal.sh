cd "$(dirname $0)"

python ../SEAL_OGB/seal_link_pred.py \
--dataset ogbl-vessel \
--model DGCNN \
--device 0 \
--epochs 3 \
--lr 0.0001 \
--dynamic_train \
--dynamic_val \
--dynamic_test \
--runs 10 \
--hidden_channels 64 \
--seed 0 