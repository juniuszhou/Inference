./target/release/candle-usage-train --model_path ./model.bin \
--vocab_path ./vocab.json --train_path ./train.txt --val_path ./val.txt \
--batch_size 128 --seq_len 128 --d_model 512 --d_ff 2048 --n_heads 8 --n_layers 6 \
--dropout 0.1 --vocab_size 50000 --max_seq_len 128


