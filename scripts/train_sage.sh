cd "$(dirname $0)"

python ../main.py \
--hidden_channel 8 \
--num_layer 2 \
--dropout 0.5 \
--runs 10 \
--lr 0.000002 \
--epochs 100 \
--device 0 \
--use_sage \
--seed 0 