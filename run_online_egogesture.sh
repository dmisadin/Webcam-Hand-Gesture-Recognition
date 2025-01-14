#!/bin/bash

# "$1" classıfıer resume path
# "$2" model_clf
# "$3" width_mult
# "$4" classıfıer modalıty
python online_test.py \
	--root_path ~/\
	--video_path datasets/EgoGesture \
	--annotation_path annotation_EgoGesture/egogestureall.json \
	--resume_path_det results/shared/egogesture_resnetl_10_Depth_8.pth \
	--resume_path_clf "$1"  \
	--result_path results \
	--dataset egogesture    \
	--sample_duration_det 8 \
	--sample_duration_clf 32 \
	--model_det resnetl \
	--model_clf "$2" \
	--model_depth_det 10 \
	--width_mult_det 0.5 \
	--model_depth_clf 101 \
	--width_mult_clf "$3" \
	--resnet_shortcut_det A \
	--resnet_shortcut_clf B \
	--batch_size 1 \
	--n_classes_det 2 \
	--n_finetune_classes_det 2 \
	--n_classes_clf 83 \
	--n_finetune_classes_clf 83 \
	--n_threads 16 \
	--checkpoint 1 \
	--modality_det Depth \
	--modality_clf "$4" \
	--n_val_samples 1 \
	--train_crop random \
	--test_subset test  \
	--det_strategy median \
	--det_queue_size 4 \
	--det_counter 2 \
	--clf_strategy median \
	--clf_queue_size 32 \
	--clf_threshold_pre 1.0 \
	--clf_threshold_final 0.15 \
	--stride_len 1 \

#python online_test.py --root_path 'D:/FESB/zavrsni_rad/Real-time-GesRec' --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/egogestureall.json --resume_path_det results/shared/egogesture_resnetl_10_Depth_8.pth --resume_path_clf "results/shared/egogesture_resnext_101_Depth_32.pth" --result_path results --dataset egogesture --sample_duration_det 8 --sample_duration_clf 32 --model_det resnetl --model_clf "resnext" --model_depth_det 10 --width_mult_det 0.5 --model_depth_clf 101 --resnet_shortcut_det A --resnet_shortcut_clf B --batch_size 1 --n_classes_det 2 --n_finetune_classes_det 2 --n_classes_clf 83 --n_finetune_classes_clf 83 --n_threads 16 --checkpoint 1 --modality_det Depth --modality_clf Depth --n_val_samples 1 --train_crop random --test_subset test --det_strategy median --det_queue_size 4 --det_counter 2 --clf_strategy median --clf_queue_size 32 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1 

bash run_online_egogesture.sh results/EgoGesture_Depth_All_but_None/egogesture_mobilenet_2.0x_Depth_16_best.pth mobilenet 2.0 Depth

python online_test.py --root_path 'D:/FESB/zavrsni_rad/Real-time-GesRec' --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/depth/egogestureall.json --resume_path_det results/shared/egogesture_resnetl_10_Depth_8.pth --resume_path_clf "results/EgoGesture_Depth_all/egogesture_mobilenet_2.0x_Depth_16_best.pth" --result_path results --dataset egogesture --sample_duration_det 8 --sample_duration_clf 16 --model_det resnetl --model_clf "mobilenet" --model_depth_det 10 --width_mult_det 0.5 --model_depth_clf 101 --width_mult_clf 2.0 --resnet_shortcut_det A --resnet_shortcut_clf B --batch_size 1 --n_classes_det 2 --n_finetune_classes_det 2 --n_classes_clf 84 --n_finetune_classes_clf 84 --n_threads 16 --checkpoint 1 --modality_det Depth --modality_clf Depth --n_val_samples 1 --train_crop random --test_subset test --det_strategy median --det_queue_size 8 --det_counter 2 --clf_strategy median --clf_queue_size 16 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1 


## RGB batch_size zna radit problem, sa 1 radi
python online_test.py --root_path 'D:/FESB/zavrsni_rad/Real-time-GesRec' --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/rgb/egogestureall.json --resume_path_det results/shared/egogesture_resnetl_10_RGB_8.pth --resume_path_clf "results/EgoGesture_RGB_all/egogesture_mobilenet_2.0x_RGB_16_best.pth" --result_path results --dataset egogesture --sample_duration_det 8 --sample_duration_clf 16 --model_det resnetl --model_clf "mobilenet" --model_depth_det 10 --width_mult_det 0.5 --model_depth_clf 101 --width_mult_clf 2.0 --resnet_shortcut_det A --resnet_shortcut_clf B --batch_size 1 --n_classes_det 2 --n_finetune_classes_det 2 --n_classes_clf 84 --n_finetune_classes_clf 84 --n_threads 16 --checkpoint 1 --modality_det 'RGB' --modality_clf RGB --n_val_samples 1 --train_crop random --test_subset test --det_strategy median --det_queue_size 8 --det_counter 2 --clf_strategy median --clf_queue_size 16 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1 


## sluzbeni klasifikator
python online_test.py --root_path "D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition" --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/rgb/egogestureall_but_None.json --resume_path_det "results/shared/egogesture_resnetl_10_RGB_8.pth" --resume_path_clf "results/shared/egogesture_resnext_101_RGB_32.pth" --result_path results/online_test/ego --dataset egogesture --sample_duration_det 8 --sample_duration_clf 32 --model_det resnetl --model_clf resnext --model_depth_det 10 --width_mult_det 1.0 --model_depth_clf 101 --width_mult_clf 1.0  --resnet_shortcut_det A --resnet_shortcut_clf B --batch_size 1 --n_classes_det 2 --n_finetune_classes_det 2 --n_classes_clf 83 --n_finetune_classes_clf 83 --n_threads 16 --checkpoint 1 --modality_det RGB --modality_clf RGB --n_val_samples 1 --train_crop random --test_subset test --det_strategy median --det_queue_size 8 --det_counter 2 --clf_strategy median --clf_queue_size 32 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1

python online_test.py --root_path "D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition" --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/rgb/egogestureall.json --resume_path_det "results/shared/egogesture_resnetl_10_RGB_8.pth" --resume_path_clf "results/eg_transfer/11/egogesture_resnext_1.0x_RGB_32_best.pth" --result_path results/online_test/ego --dataset egogesture --sample_duration_det 8 --sample_duration_clf 32 --model_det resnetl --model_clf resnext --model_depth_det 10 --width_mult_det 1.0 --model_depth_clf 101 --width_mult_clf 1.0  --resnet_shortcut_det A --resnet_shortcut_clf B --batch_size 1 --n_classes_det 2 --n_finetune_classes_det 2 --n_classes_clf 84 --n_finetune_classes_clf 84 --n_threads 16 --checkpoint 1 --modality_det RGB --modality_clf RGB --n_val_samples 1 --train_crop random --test_subset test --det_strategy median --det_queue_size 8 --det_counter 2 --clf_strategy median --clf_queue_size 32 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1

## NVgesture
python online_test.py --root_path "D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition" --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/rgb/egogestureall_but_None.json --resume_path_det "results/shared/egogesture_resnetl_10_RGB_8.pth" --resume_path_clf "results/shared/egogesture_resnext_101_RGB_32.pth" --result_path results/online_test/ego --dataset egogesture --sample_duration_det 8 --sample_duration_clf 32 --model_det resnetl --model_clf resnext --model_depth_det 10 --width_mult_det 1.0 --model_depth_clf 101 --width_mult_clf 1.0  --resnet_shortcut_det A --resnet_shortcut_clf B --batch_size 1 --n_classes_det 2 --n_finetune_classes_det 2 --n_classes_clf 83 --n_finetune_classes_clf 83 --n_threads 16 --checkpoint 1 --modality_det RGB --modality_clf RGB --n_val_samples 1 --train_crop random --test_subset test --det_strategy median --det_queue_size 8 --det_counter 2 --clf_strategy median --clf_queue_size 32 --clf_threshold_pre 1.0 --clf_threshold_final 0.15 --stride_len 1