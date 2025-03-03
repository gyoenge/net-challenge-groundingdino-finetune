from groundingdino.util.inference import load_model
from groundingdino.util.evalutation import (
    load_config, 
    measure_speed, 
    predictions_groundtruths, 
    calculate_confusion_matrix, 
    calculate_precision_recall, 
    calculate_ap50, 
    calculate_ap50_95, 
    plot_confusion_matrix, 
    save_visualized_results
)


# config 
config = load_config(config_path="evaluation_config.yaml")


def eval(config):
    model = load_model(config['model']['config_path'], config['model']['weight_path'])

    # speed 
    if config['evaluation']['speed']: 
        # result will be printed 
        total_time = measure_speed(
            model, 
            image_path=config['dataset']['image_path'], 
            text_prompt=config['evaluation']['prompt'], 
            logging=False
        )
        print(f"Total time for predictions: {total_time:.2f} seconds")

    # AP
    if config['evaluation']['ap']: 
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
        ## precision, recall, AP50, AP50-95
        print(f"precision: {precision}, recall: {recall}, AP50: {ap50}, AP50-95: {ap50_95}")
        ## plot confusion matrix 
        plot_confusion_matrix(confusion_matrix)

    # visualize result 
    if config['evaluation']['visualize_prediction']: 
        save_visualized_results(
            model, 
            image_path=config['dataset']['image_path'], 
            label_path=config['dataset']['prompt'], 
            result_path=config['evaluation']['save_path'],
            text_prompt=config['evaluation']['prompt'], 
            box_threshold=config['evaluation']['box_threshold'], 
            text_threshold=config['evaluation']['text_threshold'],
        )


if __name__ == "__main__":
    eval(config=config) 

