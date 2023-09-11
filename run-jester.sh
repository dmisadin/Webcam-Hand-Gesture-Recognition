
python main.py --root_path ~/ \
	--video_path ~/datasets/jester \
	--annotation_path Efficient-3DCNNs/annotation_Jester/jester.json \
	--result_path Efficient-3DCNNs/results \
	--dataset jester \
	--n_classes 27 \
	--model mobilenet \
	--groups 3 \
	--width_mult 0.5 \
	--train_crop random \
	--learning_rate 0.1 \
	--sample_duration 16 \
	--downsample 2 \
	--batch_size 16 \
	--n_threads 16 \
	--checkpoint 1 \
	--n_val_samples 1 \
	# --no_train \
 	# --no_val \
 	# --test


python main.py --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --video_path datasets/jester --annotation_path annotation_Jester/jester.json --result_path results/jester --dataset jester --n_classes 27 --n_finetune_classes 27 --model resnext --model_depth 101 --resnet_shortcut B --n_epochs 30  --width_mult 1 --train_crop random --learning_rate 0.04 --sample_duration 32 --batch_size 16 --n_threads 1 --checkpoint 1 --n_val_samples 1 --modality RGB #--groups 3 