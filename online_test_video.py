import os
import json
import numpy as np
import torch
import time
from torch.autograd import Variable
from PIL import Image
import cv2
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from utils import Queue

import pdb
import numpy as np

import win32api # https://learn.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
from win32con import VK_MEDIA_PLAY_PAUSE, KEYEVENTF_EXTENDEDKEY, VK_MEDIA_NEXT_TRACK, VK_MEDIA_PREV_TRACK, VK_CONTROL, KEYEVENTF_KEYUP, VK_SNAPSHOT

# EgoGesture labels, 84 = 83 classes + 'None'
labels=["Scroll_hand_towards_right", "Scroll_hand_towards_left", "Scroll_hand_downward", "Scroll_hand_upward", "Scroll_hand_forward", "Scroll_hand_backward", "Cross_index_fingers", "Zoom_in_with_fists", "Zoom_out_with_fists", "Rotate_fists_clockwise", "Rotate_fists_counterclockwise", "Zoom_in_with_fingers", "Zoom_out_with_fingers", "Rotate_fingers_clockwise", "Rotate_fingers_counterclockwise", "Click_with_index_finger", "Sweep_diagonal", "Sweep_circle", "Sweep_cross", "Sweep_checkmark", "Static_fist", "Measure(distance)", "Photo_frame", "Number_0", "Number_1", "Number_2", "Number_3", "Number_4", "Number_5", "Number_6", "Number_7", "Number_8", "Number_9", "OK", "Another_number_3", "Pause", "Shape_C", "Make_a_phone_call", "Wave_hand", "Wave_finger", "Knock", "Beckon", "Palm_to_fist", "Fist_to_Palm", "Trigger_with_thumb", "Trigger_with_index_finger", "Hold_fist_in_the_other_hand", "Grasp", "Walk", "Gather_fingers", "Snap_fingers", "Applaud", "Dual_hands_heart", "Put_two_fingers_together", "Take_two_fingers_apart", "Turn_over", "Move_fist_upward", "Move_fist_downward", "Move_fist_toward_left", "Move_fist_toward_right", "Bring_hand_close", "Push_away", "Thumb_upward", "Thumb_downward", "Thumb_toward_right", "Thumb_toward_left", "Thumbs_backward", "Thumbs_forward", "Move_hand_upward", "Move_hand_downward", "Move_hand_towards_left", "Move_hand_towards_right", "Draw_circle_with_hand_in_horizontal_surface", "Bent_number_2", "Bent_another_number_3", "Dual_fingers_heart", "Scroll_fingers_toward_left", "Scroll_fingers_toward_right", "Move_fingers_upward", "Move_fingers_downward", "Move_fingers_left", "Move_fingers_right", "Move_fingers_forward", "None"]

#NVGesture labels, 26 
#labels = ["Move hand left", "Move hand right", "Move hand up", "Move hand down", "Move two fingers left", "Move two fingers right", "Move two fingers up", "Move two fingers down", "Click index finger", "Call someone", "Open hand", "Shaking hand", "Show index finger", "Show two fingers", "Show three fingers", "Push hand up", "Push hand down", "Push hand out", "Pull hand in", "Rotate fingers CW", "Rotate figners CCW", "Push two fingers away", "Close hand two times", "Thumb up", "Okay gesture"]

###Pretrained RGB models
##Google Drive
#https://drive.google.com/file/d/1V23zvjAKZr7FUOBLpgPZkpHGv8_D-cOs/view?usp=sharing

'''
Where j is the iteration index of an active state, at which a
gesture is detected, and t is calculated as 9
'''
def weighting_func(j):
    return (1 / (1 + np.exp(-0.2 * (j - 9))))

def executeAction(actions): 
    print(f"Actions: {actions}")
    for action in actions:
        win32api.keybd_event(keymap[action], 0, 0, 0)
    time.sleep(0.05)
    for action in actions:
        win32api.keybd_event(keymap[action], 0, KEYEVENTF_KEYUP, 0)

opt = parse_opts_online()


def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.width_mult = opt.width_mult_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = opt.n_classes_det
    opt.n_finetune_classes = opt.n_finetune_classes_det

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)
    detector = detector.cuda()
    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        detector.load_state_dict(checkpoint['state_dict'], strict=False)

    print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.width_mult = opt.width_mult_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)
    classifier = classifier.cuda()
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier


detector, classifier = load_models(opt)

if opt.no_mean_norm and not opt.std_norm:
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not opt.std_norm:
    norm_method = Normalize(opt.mean, [1, 1, 1])
else:
    norm_method = Normalize(opt.mean, opt.std)

spatial_transform = Compose([
    Scale(112),
    CenterCrop(112),
    ToTensor(opt.norm_value), norm_method
])

opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)
fps = ""
prev_frame_time = 0
new_frame_time = 0
#cap = cv2.VideoCapture(opt.video)
cap = cv2.VideoCapture(0)
num_frame = 0
clip = []
active_index = 0
passive_count = 0
active = False
prev_active = False
finished_prediction = None
pre_predict = False
detector.eval()
classifier.eval()
cum_sum = np.zeros(opt.n_classes_clf, )
det_selected_queue = np.zeros(opt.n_classes_det, )
clf_selected_queue = np.zeros(opt.n_classes_clf, )
myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
results = []
prev_best1 = opt.n_classes_clf
spatial_transform.randomize_parameters()
result_count = 0
prev_result_count = 0
predicted = []


## Start GUI configuration and save key binds:
import gui.gui as gui
gestures = gui.config
print(gestures)

with open("gui/keymap.json", "r") as read_file:
    keymap = json.load(read_file)

while cap.isOpened():
    new_frame_time = time.time()
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1) # Flip the image if the model was trained on Egocentric footage
       
    # Check if a frame was read successfully
    if not ret:
        print("End of video.")
        break

    # Check if the frame has valid dimensions before resizing
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Invalid frame detected. Skipping.")
        continue

    if num_frame == 0:
        cur_frame = cv2.resize(frame,(320,240))
        cur_frame = Image.fromarray(cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB))
        cur_frame = cur_frame.convert('RGB')
        for i in range(opt.sample_duration):
            clip.append(cur_frame)
        clip = [spatial_transform(img) for img in clip]

    clip.pop(0)
    _frame = cv2.resize(frame,(320,240))
    _frame = Image.fromarray(cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB))
    _frame = _frame.convert('RGB')
    _frame = spatial_transform(_frame)
    clip.append(_frame)
    im_dim = clip[0].size()[-2:]

    try:
        test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
    except Exception as e:
        pdb.set_trace()
        raise e
    
    inputs = torch.cat([test_data],0).view(1,3,opt.sample_duration,112,112)
    num_frame += 1

    ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
    with torch.no_grad():
        inputs = Variable(inputs)
        inputs_det = inputs[:, :, -opt.sample_duration_det:, :, :]
        outputs_det = detector(inputs_det)
        outputs_det = F.softmax(outputs_det, dim=1)
        outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
        # enqueue the probabilities to the detector queue
        myqueue_det.enqueue(outputs_det.tolist())

        if opt.det_strategy == 'raw':
            det_selected_queue = outputs_det
        elif opt.det_strategy == 'median':
            det_selected_queue = myqueue_det.median
        elif opt.det_strategy == 'ma':
            det_selected_queue = myqueue_det.ma
        elif opt.det_strategy == 'ewma':
            det_selected_queue = myqueue_det.ewma
        prediction_det = np.argmax(det_selected_queue)

        prob_det = det_selected_queue[prediction_det]
        
        #### State of the detector is checked here as detector act as a switch for the classifier
        if prediction_det == 1:
            inputs_clf = inputs[:, :, :, :, :]
            inputs_clf = torch.Tensor(inputs_clf.numpy()[:,:,::1,:,:])
            outputs_clf = classifier(inputs_clf)
            outputs_clf = F.softmax(outputs_clf, dim=1)
            outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
            # Push the probabilities to queue
            myqueue_clf.enqueue(outputs_clf.tolist())
            passive_count = 0

            if opt.clf_strategy == 'raw':
                clf_selected_queue = outputs_clf
            elif opt.clf_strategy == 'median':
                clf_selected_queue = myqueue_clf.median
            elif opt.clf_strategy == 'ma':
                clf_selected_queue = myqueue_clf.ma
            elif opt.clf_strategy == 'ewma':
                clf_selected_queue = myqueue_clf.ewma

        else:
            outputs_clf = np.zeros(opt.n_classes_clf, )
            # Push the probabilities to queue
            myqueue_clf.enqueue(outputs_clf.tolist())
            passive_count += 1
    
    if passive_count >= opt.det_counter:
        active = False
    else:
        active = True

    # one of the following line need to be commented !!!!
    if active:
        active_index += 1
        cum_sum = ((cum_sum * (active_index - 1)) + (weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
        #cum_sum = ((cum_sum * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach
        best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
        if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
            finished_prediction = True
            pre_predict = True
    else:
        active_index = 0

    if active == False and prev_active == True:
        finished_prediction = True
    elif active == True and prev_active == False:
        finished_prediction = False

    if finished_prediction == True:
        #print(finished_prediction,pre_predict)
        best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
        if cum_sum[best1] > opt.clf_threshold_final:
            if pre_predict == True:
                if best1 != prev_best1:
                    if cum_sum[best1] > opt.clf_threshold_final:
                        results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                        print('Early Detected - class : {} with prob : {} at frame {}'
                              .format(best1, cum_sum[best1], ( i * opt.stride_len) + opt.sample_duration_clf))
            else:
                if cum_sum[best1] > opt.clf_threshold_final:
                    if best1 == prev_best1:
                        if cum_sum[best1] > 5:
                            results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                            print('Late Detected - class : {} with prob : {} at frame {}'
                                  .format(best1, cum_sum[best1], (i * opt.stride_len) + opt.sample_duration_clf))
                    else:
                        results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
                        print('Late Detected - class : {} with prob : {} at frame {}'
                              .format(best1, cum_sum[best1], (i * opt.stride_len) + opt.sample_duration_clf))

            finished_prediction = False
            prev_best1 = best1

        cum_sum = np.zeros(opt.n_classes_clf, )
    
    if active == False and prev_active == True:
        pre_predict = False

    prev_active = active
    elapsedTime = new_frame_time - prev_frame_time
    
    fps = "(Playback) {:.1f} FPS".format(1 / elapsedTime)

    prev_frame_time = new_frame_time

    result_count = len(results)

    if len(results) != 0:
        #predicted = np.array(results)[:, 1]
        prev_best1 = -1
        if result_count > prev_result_count:
            predicted.append(results[-1][1])
            print('predicted classes: \t', predicted)  
        cv2.putText(frame, labels[results[-1][1]], (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 2, (38, 0, 255), 2, cv2.LINE_AA)
    else:
        predicted = []

    prev_result_count = result_count

    #0xBB plus, 0xBD minus

    if len(predicted) != 0:
        current = predicted.pop()

        #treba nesto kao .find() u javascriptu za dobit actions array
        actionList = []
        temp_list = list(filter(lambda item: item.get("class") == current, gui.config))
        if len(temp_list) > 0:
            actionList = temp_list[0]["action"]
            print(list(filter(lambda item: item.get("class") == current, gui.config)))
            executeAction(actionList)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('User exit.')
        break

""" for r in results:
    print(labels[r[1]]) 
print(results) """
cap.release()
cv2.destroyAllWindows()


