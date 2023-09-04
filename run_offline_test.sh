python offline_test.py --root_path "D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition" --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/rgb/egogestureall.json  --resume_path "results/eg_transfer/11/egogesture_resnext_1.0x_RGB_32_best.pth" --result_path results/online_test/ego --dataset egogesture --sample_duration 32 --model resnext --model_depth 101  --resnet_shortcut B --batch_size 8 --n_classes 84 --n_finetune_classes 84 --modality RGB --test_subset training


#--resume_path_det "results/shared/egogesture_resnetl_10_RGB_8.pth" 
#--resume_path_clf "results/shared/egogesture_resnext_101_RGB_32.pth" 


python offline_test.py --root_path "D:/FESB/zavrsni_rad/Webcam-Hand-Gesture-Recognition" --video_path datasets/EgoGesture --annotation_path annotation_EgoGesture/rgb/egogestureall_but_None.json  --resume_path "results/shared/egogesture_resnext_101_RGB_32.pth" --result_path results/online_test/shared --dataset egogesture --sample_duration 32 --model resnext --model_depth 101  --resnet_shortcut B --batch_size 8 --n_classes 83 --n_finetune_classes 83 --modality RGB --test_subset val