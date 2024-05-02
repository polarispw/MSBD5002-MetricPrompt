python -W ignore ../main.py \
  --dataset "agnews" \
  --k_shot 2 \
  --n_adapt_epochs 120 \
  --prompt_template 0 \
  --pivot 0 \
  --start_episode 0 \
  --num_episodes 10 \
  --kernl_accerleration 0 \
  --seed 1999 \
  --data_path "data" \
  --output_die "./output_dir/" \
  --pretrained_model "bert" \
  --model_type "bert-base-uncased" \
  --num_episodes 10 \
  --pooling "mean" \
  
  
