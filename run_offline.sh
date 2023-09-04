#!/bin/bash
python3 offline_test.py \
	--root_path D:/FESB/zavrsni rad/Real-time-GesRec \
	--video_path D:/FESB/zavrsni rad/nvgesture_arch \
	--annotation_path annotation_nvGesture/nvall.json \
	--resume_path_det report/egogesture_resnetl_10_Depth_8_9939.pth \
	--resume_path_clf report/egogesture_resnext_101_Depth_32_9403.pth  \
	--result_path results \
	--dataset nvgesture    \
	--sample_duration_det 8 \
	--sample_duration_clf 32 \
	--model_det resnetl \
	--model_clf resnext \
	--model_depth_det 10 \
	--model_depth_clf 101 \
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
	--modality_clf Depth \
	--n_val_samples 1 \
	--train_crop random \
	--test_subset test  \
	--det_strategy median \
	--det_queue_size 4 \
	--det_counter 2 \
	--clf_strategy median \
	--clf_queue_size 16 \
	--clf_threshold_pre 0.6 \
	--clf_threshold_final 0.15 \
	--stride_len 1 \
	--no_cuda \

python main.py --video_path 'D:/FESB/zavrsni rad/nvgesture_arch/Video_data' --dataset nvgesture --root_path 'D:/FESB/zavrsni rad/Real-time-GesRec' --annotation_path 'annotation_nvGesture/nvall.json' --no_cuda

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:\FESB\zavrsni_rad\Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --pretrain_path 'results/shared/egogesture_resnext_101_RGB_32.pth' --pretrain_modality RGB --batch_size 16 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 10 --modality RGB --n_classes 25 --n_finetune_classes 25 --train_crop random

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:\FESB\zavrsni_rad\Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --batch_size 10 --sample_duration 8 --model resnet --model_depth 10  --n_epochs 10 --modality Depth --n_classes 25 --n_finetune_classes 25 --train_crop random --no_val


python main.py --video_path 'datasets/EgoGesture/images' --dataset egogesture --root_path 'D:\FESB\zavrsni_rad\Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_EgoGesture/egogestureall.json' --result_path "results/ego_clf" --batch_size 10 --sample_duration 32 --model resnext --model_depth 101 --n_epochs 10 --modality Depth --n_classes 83 --n_finetune_classes 83 --train_crop random

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:\FESB\zavrsni_rad\Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvbinary.json' --result_path "results/nv_new" --batch_size 8 --sample_duration 8 --model resnetl --model_depth 10 --n_epochs 10 --modality RGB --n_classes 2 --n_finetune_classes 2 --train_crop random --width_mult 1 --resnet_shortcut A

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:\FESB\zavrsni_rad\Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --pretrain_path 'results/shared/egogesture_resnext_101_RGB_32.pth' --pretrain_modality RGB --batch_size 16 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 10 --modality RGB --n_classes 83 --n_finetune_classes 26 --train_crop random --resnet_shortcut B

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:\FESB\zavrsni_rad\Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --result_path "results/nv_clean" --batch_size 16 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 50 --modality RGB --n_classes 26 --n_finetune_classes 26 --train_crop random --resnet_shortcut B --learning_rate 0.02

#Transfer learning jester -> nvgesture

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --result_path "results/nv_transfer/jester-nv" --batch_size 8 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 27 --n_finetune_classes 26 --train_crop random --resnet_shortcut B --pretrain_path 'results/shared/jester_resnext_101_RGB_32.pth' --learning_rate 0.01


#learning from scratch nvgesture
python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --result_path "results/nv_clean" --batch_size 16 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 26 --n_finetune_classes 26 --train_crop random --resnet_shortcut B --learning_rate 0.04

#Transfer learning EgoGesture -> nvgesture

python main.py --video_path 'datasets/nvgesture' --dataset nvgesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_nvGesture/nvall.json' --result_path "results/nv_transfer" --batch_size 8 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 83 --n_finetune_classes 26 --train_crop random --resnet_shortcut B --pretrain_path 'results/shared/egogesture_resnext_101_RGB_32.pth' --learning_rate 0.01

#Transfer learning jester -> egogesture
python main.py --video_path 'datasets/EgoGesture' --dataset egogesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_EgoGesture/rgb/egogestureall.json' --result_path "results/eg_transfer" --batch_size 10 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 27 --n_finetune_classes 84 --train_crop random --resnet_shortcut B --pretrain_path 'results/shared/jester_resnext_101_RGB_32.pth' --learning_rate 0.006

#Transfer learning jester-nvgesture -> egogesture
python main.py --video_path 'datasets/EgoGesture' --dataset egogesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_EgoGesture/rgb/egogestureall.json' --result_path "results/eg_transfer" --batch_size 8 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 26 --n_finetune_classes 84 --train_crop random --resnet_shortcut B --pretrain_path 'results/nv_transfer/jester-nv/2/nvgesture_resnext_1.0x_RGB_32_best.pth' --learning_rate 0.01

#Learning egogesture from scratch
### ideja, spusti learning rate na 0.01
python main.py --video_path 'datasets/EgoGesture' --dataset egogesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_EgoGesture/rgb/egogestureall.json' --result_path "results/ego_clean" --batch_size 8 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 84 --n_finetune_classes 84 --train_crop random --resnet_shortcut B --learning_rate 0.01


# Transfer ego iz rada - ego

python main.py --video_path 'datasets/EgoGesture' --dataset egogesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_EgoGesture/rgb/egogestureall_but_None.json' --result_path "results/eg_transfer" --batch_size 8 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 83 --n_finetune_classes 84 --train_crop random --resnet_shortcut B --pretrain_path 'results/egogesture_resnext_1.0x_RGB_32_checkpoint.pth' --learning_rate 0.01

python main.py --video_path 'datasets/EgoGesture' --dataset egogesture --root_path 'D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition' --annotation_path 'annotation_EgoGesture/rgb/egogestureall_but_None.json' --result_path "results/eg_transfer" --batch_size 10 --sample_duration 32 --model resnext --model_depth 101  --n_epochs 30 --modality RGB --n_classes 83 --n_finetune_classes 83 --train_crop random --resnet_shortcut B --pretrain_path 'results/shared/egogesture_resnext_101_RGB_32.pth' --learning_rate 0.001