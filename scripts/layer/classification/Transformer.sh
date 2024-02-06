export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=Transformer

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
  --task_name classification \
  --is_training 1 \
  --root_path ./dataset/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
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
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
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
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
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
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

python3 -u run.py \
  --use_multi_gpu \
  --itr 3 \
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
  --top_k 3 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10
