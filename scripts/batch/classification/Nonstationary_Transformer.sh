export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=Nonstationary_Transformer

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
  --normalization batch \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 2048 \
  --d_ff 256 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --enc_in 3

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
  --normalization batch \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --enc_in 3

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
  --normalization batch \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --enc_in 3

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
  --normalization batch \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --enc_in 3

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
  --normalization batch \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --p_hidden_dims 256 256 \
  --p_hidden_layers 2 \
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --enc_in 3
