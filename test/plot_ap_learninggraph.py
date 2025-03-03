import numpy as np
import matplotlib.pyplot as plt

from groundingdino.util.inference import load_model
from groundingdino.util.evalutation import (
    load_config, 
    predictions_groundtruths, 
    calculate_confusion_matrix, 
    calculate_precision_recall, 
    calculate_ap50, 
    calculate_ap50_95, 
)

# config 
config = load_config(config_path="evaluation_config.yaml")
MAX_EPOCH = 5
WEIGHTS_PATHS = [ f"weights/model_weights_knife_{i}.pth" for i in range(1,MAX_EPOCH+1)]

def AP_eval_for_learningGraph():
    precisions, recalls, mAP50s, mAP50_95s = [], [], [], []    
    for weights_path in WEIGHTS_PATHS:
        model = load_model(weights_path, config['model']['weight_path'])
        predictions, groundtruths = predictions_groundtruths(
            model, 
            image_path=config['dataset']['image_path'], 
            label_path=config['dataset']['prompt'], 
            text_prompt=config['evaluation']['prompt'], 
            box_threshold=config['evaluation']['box_threshold'], 
            text_threshold=config['evaluation']['text_threshold'], 
        )
        confusion_matrix = calculate_confusion_matrix(
                predictions, 
                groundtruths, 
                iou_threshold=0.5, 
                confidendce_threshold=0.5
        )
        precision, recall = calculate_precision_recall(confusion_matrix)
        ap50 = calculate_ap50(predictions, groundtruths)
        ap50_95 = calculate_ap50_95(predictions, groundtruths)
        
        precisions.append(precision)
        recalls.append(recall)
        mAP50s.append(ap50)
        mAP50_95s.append(ap50_95)

    # Plotting
    epochs_range = np.arange(MAX_EPOCH)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, precisions, '-o')
    plt.title("metrics/precision(B)")
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, recalls, '-o')
    plt.title("metrics/recall(B)")
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, mAP50s, '-o')
    plt.title("metrics/mAP50(B)")
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, mAP50_95s, '-o')
    plt.title("metrics/mAP50-95(B) ")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__": 
    AP_eval_for_learningGraph()