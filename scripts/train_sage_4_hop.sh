cd "$(dirname $0)"

python ../main.py \
--hidden_channel 8 \
--num_layer 4 \
--dropout 0.5 \
--runs 10 \
--lr 0.000001 \
--epochs 10 \
--device 3 \
--use_sage \
--seed 0 