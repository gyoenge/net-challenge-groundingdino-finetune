from groundingdino.util.inference import load_image, predict 
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
import groundingdino.datasets.transforms as T

import os, glob 
from PIL import Image 
import numpy as np
import time
import yaml 
from sklearn.metrics import auc
import matplotlib.pyplot as plt

# whether print log 
LOGGING_ON = False 

###

def load_config(config_path: str): 
    with open(config_path, 'r') as config_file: 
        config = yaml.safe_load(config_file)
    
    required_keys = {
        "model": ["config_path", "weight_path"],
        "dataset": ["image_path", "label_path"],
        "evaluation": ["prompt", "box_threshold", "text_threshold",
                       "speed", "ap", "visualize_prediction", "save_path"]
    }

    try: 
        for section, keys in required_keys.items(): 
            if section not in config: 
                raise KeyError(f"Missing Section: {section}")
            for key in keys: 
                if key not in config[section]:
                    raise KeyError(f"Missing Key: {section}.{key}")
    except Exception as e: 
        print("Failed to load evaluation config")

    return config


def load_image(image_file, newSize=None):
    if newSize is None:
        # newSize = [800]
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else: 
        transform = T.Compose(
            [
                T.RandomResize(newSize, max_size=4000),
                #T.RandomResize(newSize, max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    image_source = Image.open(image_file).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

### 

def _best_nms(
        image_source, 
        prompt, 
        boxes, 
        logits, 
        phrases, 
        box_threshold,
        nms_threshold=0.3,
):  
    ## nms per phrase
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    converted_scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    if LOGGING_ON : print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = converted_scaled_boxes[indices]
        phrase_boxes = converted_scaled_boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, nms_threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    if not nms_boxes_list or not nms_logits_list:
        if LOGGING_ON : print("[hard nms] there are no nms boxes at all.")
        nms_result = None
    else:
        if LOGGING_ON : print(f"[hard nms] there are {len(nms_boxes_list)} nms boxes.")
        nms_result = torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list
    
    ## best nms box 
    max_logit = -float('inf')
    best_box = None
    if nms_result is not None:
        boxes, logits, phrases = nms_result
        # index = torch.argmax(logits).item() # only use best box 
        for box, logit, phrase in zip(boxes, logits, phrases):
            ######if phrase == self.text_prompt and logit >= self.box_threshold:
            if logit >= box_threshold:
                if logit > max_logit:
                    max_logit = logit
                    class_true = 1 if prompt==phrase else 0 
                    # best_box : [x1, y1, x2, y2, logit, class_true]
                    best_box = torch.cat([box, torch.tensor([logit, class_true])], dim=0)
    
    if best_box is not None:     
        if LOGGING_ON : print("[hard nms] best box ([x1, y1, x2, y2, logit, class_true]) : ", best_box)
        # print(torch.tensor(boxes[index]).unsqueeze(0), torch.tensor(logits[index]).unsqueeze(0), [phrases[index]])
        return best_box
    else: 
        return torch.tensor([0, 0, 0, 0, 0, 0])


def detect_best_box(
        model, 
        image_file, 
        text_prompt, 
        box_threshold, 
        text_threshold, 
        newSize=None, 
        normalize=False, 
):
    # load image 
    image_source, image = load_image(image_file, newSize)
    file_name = os.path.basename(image_file)  # includes extension 

    # model prediction 
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # find best box 
    # best_box : [x1, y1, x2, y2, logit, class_true]
    best_box = _best_nms(
        image_source, 
        text_prompt, 
        boxes, 
        logits, 
        phrases, 
        box_threshold,
        nms_threshold=0.3
    )

    # normalize 
    if normalize: 
        h, w, _ = image_source.shape
        x1, y1, x2, y2, confidence, classtruth = best_box
        normalized = torch.tensor([x1/w, y1/h, x2/w, y2/h, confidence, classtruth])
        return normalized

    return best_box 


def predictions_groundtruths(
        model, 
        image_path, 
        label_path,
        text_prompt, 
        box_threshold, 
        text_threshold, 
):
    # boxes should be normalized 

    predictions, groundtruths = [], []   
    images_files = glob.glob(os.path.join(image_path, '*'))        
    for idx, image_file in enumerate(images_files):
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_path, file_name + ".txt")
        if not os.path.exists(label_file):  
            continue 
        prediction = detect_best_box(
            model, 
            image_file, 
            text_prompt, 
            box_threshold, 
            text_threshold, 
            newSize=None, 
            normalized=True,
        )
        image_source, _ = load_image(image_file, newSize=None)
        groundtruth = read_label_txt(image_source, label_file, normalized=True)

        predictions.append(prediction)
        groundtruths.append(groundtruth)
    
    return predictions, groundtruths


### 

def read_label_txt(image_source, label_file, normalized=True):
    """
    read single label 
    """
    # cxcywh 형식 
    with open(label_file, 'r') as f:
        content = f.readline().strip()
        values = content.split()
        boxes = torch.Tensor([float(values[1]), float(values[2]), float(values[3]), float(values[4])])
        # unnormalize ?
        if not normalized: 
            h, w, _ = image_source.shape
            scaled_boxes = boxes * torch.Tensor([w, h, w, h])
            label_box = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        else: 
            label_box = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    return label_box   


def save_visualized_result(
        image_file, 
        label_file, 
        result_path, 
        best_box, # normalized=False 
): 
    """
    save single visulization  
    """
    # load image 
    image_source, image = load_image(image_file, newSize=None)
    file_name = os.path.basename(image_file)  # includes extension 

    # read label 
    groundtruth = read_label_txt(image_source, label_file, normalized=False)

    # visualize result 
    image = cv2.imread(image_file)
    _bb_x1, _bb_y1, _bb_x2, _bb_y2, bb_logit, _ = best_box
    bb_x1, bb_y1, bb_x2, bb_y2 = map(int, [_bb_x1, _bb_y1, _bb_x2, _bb_y2])
    if LOGGING_ON : print("prediction : ", bb_x1, bb_y1, bb_x2, bb_y2)
    gt_x1, gt_y1, gt_x2, gt_y2 = map(int, groundtruth)
    if LOGGING_ON : print("groundtruth : ", gt_x1, gt_y1, gt_x2, gt_y2)
    # draw prediction bbox (red)
    cv2.rectangle(image, (bb_x1, bb_y1), (bb_x2, bb_y2),  (0, 0, 255), 2)
    label_position = (bb_x1, bb_y2-10)
    cv2.putText(image, f"{bb_logit}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    # draw groundtruth bbox (blue)
    cv2.rectangle(image, (gt_x1, gt_y1), (gt_x2, gt_y2), (255, 0, 0), 2)

    # save result 
    result_file = os.path.join(result_path, file_name)
    cv2.imwrite(result_file, image)

    return result_file


def save_visualized_results(
        model, 
        image_path, 
        label_path, 
        result_path, 
        text_prompt, 
        box_threshold, 
        text_threshold, 
): 
    """
    save all visulization in path
    """  
    # boxes shoud not be normalized 

    images_files = glob.glob(os.path.join(image_path, '*'))        
    for idx, image_file in enumerate(images_files):
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        label_file = os.path.join(label_path, file_name + ".txt")
        if not os.path.exists(label_file):  
            continue 
        best_box = detect_best_box(
            model, 
            image_file, 
            text_prompt, 
            box_threshold, 
            text_threshold, 
            newSize=None, 
            normalized=False,
        )
        try: 
            result_file = save_visualized_result(
                image_file, 
                label_file, 
                result_path, 
                best_box, 
            )
            print(f"result saved : {result_file}.")
        except Exception as e:
            print(e)
    print("** All results saved **")
    
###

def measure_speed(model, image_path, text_prompt, newSize=None, logging=False): 
    """
    Measure Speed of predicting all images in image_path 
    """

    image_files = glob.glob(os.path.join(image_path, '*'))        
    start_time = time.time() # start 
    for idx, image_file in enumerate(image_files):
        print(idx+1, end=" .. ")
        best_box = detect_best_box(
            model, 
            image_file, 
            text_prompt, 
            box_threshold=0.6, 
            text_threshold=0.4, 
            newSize=newSize, 
            normalized=False, 
        )
        if logging and (idx+1)%100 == 0 :
            print(f"{idx+1}번째, {start_time-time.time()} 경과")
    end_time = time.time()
    total_time = end_time - start_time
    if logging: 
        print(f"\nTotal time for predictions: {total_time:.2f} seconds")
        print(f"총 predict 갯수 : {len(image_files)}")
    return total_time 


###

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union for two bounding boxes.
    Args:
        box1 : [x1, y1, x2, y2]
        box2 : [x1, y1, x2, y2]
    Returns:
        IoU value (float)
    """
    # Calculate the intersection coordinates
    x1, y1 = torch.max(box1[0:2], box2[0:2])
    x2, y2 = torch.min(box1[2:4], box2[2:4])

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - intersection
    iou = intersection / union
    return iou


def calculate_confusion_matrix(
        predictions, 
        groundtruths, 
        iou_threshold=0.5, 
        confidendce_threshold=0.5
):
    """
    Compute Average Precision for a single class.
    Args:
        predictions (list of tensors): List of predicted boxes [x1, y1, x2, y2, score]
        ground_truths (list of tensors): List of ground-truth boxes [x1, y1, x2, y2]
        iou_threshold (float): Threshold for IoU overlap
    Returns:
        # float: Average Precision (AP) value 
        tp / fp / tn / fn values 
    """
    
    # Sort predictions based on scores
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)

    tp = torch.zeros(len(predictions))
    fp = torch.zeros(len(predictions))
    tn = torch.zeros(len(predictions)) 
    fn = torch.zeros(len(predictions)) 
    
    for i, prediction, groudntruth in enumerate(zip(predictions, groundtruths)):
        iou = calculate_iou(prediction[0:4], groudntruth)
        if iou >= iou_threshold:
            if prediction[5] >= confidendce_threshold:
                tp[i] = 1 
            else: 
                fp[i] = 1 
        else: 
            if prediction[5] >= confidendce_threshold:
                tn[i] = 1 
            else: 
                fn[i] = 1 

    TP = torch.cumsum(tp, dim=0)
    FP = torch.cumsum(fp, dim=0)
    TN = torch.cumsum(tn, dim=0)
    FN = torch.cumsum(fn, dim=0)

    confution_matrix = { "TP": TP, "FP": FP, "FN": FN, "TN": TN } 

    return confution_matrix


def calculate_precision_recall(confusion_matrix):
    TP, FP, FN, TN = confusion_matrix["TP"], confusion_matrix["FP"], confusion_matrix["FN"], confusion_matrix["TN"]
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    return precision, recall

def calculate_precision_recall_with_thresholds(
        predictions, 
        groundtruths, 
        iou_threshold=0.5, 
        confidence_threshold=0.5
):
    confusion_matrix = calculate_confusion_matrix(
        predictions, 
        groundtruths, 
        iou_threshold, 
        confidence_threshold
    )
    precision, recall = calculate_precision_recall(confusion_matrix)
    return precision, recall

def calculate_ap(predictions, groundtruths, iou_threshold):
    # precision recall graph space 
    precisions = []
    recalls = []
    confidence_thresholds = torch.linspace(0, 1, 100) # confidence threshold steps 
    for confidence_threshold in confidence_thresholds:
        precision, recall = calculate_precision_recall_with_thresholds(
            predictions, 
            groundtruths, 
            iou_threshold=iou_threshold, 
            confidence_threshold=confidence_threshold
        )
        precisions.append(precision)
        recalls.append(recall)
    
    # calculate AP 
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    ## ensure recall is sorted in increasing order
    sorted_indices = np.argsort(recalls)  
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]
    ## make Precision monotonically decreasing
    for i in range(len(sorted_precisions) - 2, -1, -1):
        sorted_precisions[i] = max(sorted_precisions[i], sorted_precisions[i + 1])
    ## compute ap
    # ap = auc(sorted_recalls, sorted_precisions)
    ap = sum((sorted_recalls[i] - sorted_recalls[i - 1]) * sorted_precisions[i] for i in range(1, len(sorted_recalls)))
    
    return ap


def calculate_ap50(predictions, groundtruths): 
    ap50 = calculate_ap(predictions, groundtruths, iou_threshold=0.5)
    return ap50


def calculate_ap50_95(predictions, groundtruths): 
    iou_thresholds = torch.arange(0.5, 1.0, 0.05)  # 0.5~0.95 0.05 간격 
    aps = [calculate_ap(predictions, groundtruths, iou_threshold) for iou_threshold in iou_thresholds]
    ap50_95 = torch.mean(torch.tensor(aps)) 
    return ap50_95


### 

def plot_confusion_matrix(confusion_matrix):
    TP, FP, FN, TN = confusion_matrix["TP"], confusion_matrix["FP"], confusion_matrix["FN"], confusion_matrix["TN"]

    name = [["TP","FP"],["FN","TN"]]
    # Create the confusion matrix
    cm = np.array([[TP, FP], [FN, TN]])

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues")
    #cax = ax.matshow(cm, cmap="Blues", vmin=0, vmax=250)

    # Add a colorbar to the right of the matrix
    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    #cbar.ax.set_ylabel('Counts', rotation=270, labelpad=15, fontsize=14)

    # Display the values on the matrix
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{name[i][j]}: {cm[i, j]}", va='center', ha='center', fontsize=20, color="white" if cm[i,j] > (TP+TN)/2 else "black")

    ax.set_xlabel("Predicted", fontsize=14, labelpad=2)
    ax.set_ylabel("Actual", fontsize=14, rotation=90, labelpad=2)
    ax.xaxis.set_ticks_position('bottom')  
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['SmokingPerson', 'Not SmokingPerson'])
    ax.set_yticklabels(['SmokingPerson', 'Not SmokingPerson'], rotation=90, va="center")
    # ax.set_xticklabels(['Knife', 'Not Knife'])
    # ax.set_yticklabels(['Knife', 'Not Knife'], rotation=90, va="center")
    ax.set_title("Confusion Matrix", fontsize=16)

    plt.show()