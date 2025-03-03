from groundingdino.util.train import load_config, load_model, load_image, freeze_layers, train_image, annotate
import cv2
import os
import json
import csv
import torch
from collections import defaultdict
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


# config 
config = load_config(config_path="train_config.yaml")

# Model
model = load_model(config['model']['config_path'], config['model']['weight_path'])

# Dataset paths
images_files=sorted(os.listdir(config['dataset']['image_path']))
ann_file=config['dataset']['ann_path']

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (numpyarray): input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)

def draw_box_with_label(image, output_path, coordinates, label, color=(0, 0, 255), thickness=2, font_scale=0.5):
    """
    Draw a box and a label on an image using OpenCV.

    Parameters:
    - image (str):  Input image.
    - output_path (str): Path to save the image with the box and label.
    - coordinates (tuple): A tuple (x1, y1, x2, y2) indicating the top-left and bottom-right corners of the box.
    - label (str): The label text to be drawn next to the box.
    - color (tuple, optional): Color of the box and label in BGR format. Default is red (0, 0, 255).
    - thickness (int, optional): Thickness of the box's border. Default is 2 pixels.
    - font_scale (float, optional): Font scale for the label. Default is 0.5.
    """
    
    # Draw the rectangle
    cv2.rectangle(image, (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3]), color, thickness)
    
    # Define a position for the label (just above the top-left corner of the rectangle)
    label_position = (coordinates[0], coordinates[1]-10)
    
    # Draw the label
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    # Save the modified image
    cv2.imwrite(output_path, image)



def read_dataset(ann_file):
    ann_Dict= defaultdict(lambda: defaultdict(list))
    with open(ann_file) as file_obj:
        ann_reader= csv.DictReader(file_obj)  
        # Iterate over each row in the csv file
        # using reader object
        for row in ann_reader:
            #print(row)
            img_n=os.path.join("multimodal-data/images",row['image_name'])
            x1=int(row['bbox_x'])
            y1=int(row['bbox_y'])
            x2=x1+int(row['bbox_width'])
            y2=y1+int(row['bbox_height'])
            label=row['label_name']
            ann_Dict[img_n]['boxes'].append([x1,y1,x2,y2])
            ann_Dict[img_n]['captions'].append(label)
    return ann_Dict


def train(model, 
          ann_file, 
          epochs=1, 
          learning_rates=None,
          save_path='weights/model_weights',
          save_epoch=50
    ):
    
    # Read Dataset
    ann_Dict = read_dataset(ann_file)
    
    # layer-wise learning rate 
    if learning_rates is None:
        learning_rates = {
            "backbone": 1e-6,
            "transformer": 1e-5,
            "bert": 5e-6,
            "feat_map": 1e-5,
            "input_proj": 1e-5,
            "bbox_embed": 1e-4,
            "class_embed": 1e-4
        }

    # Add optimizer
    optimizer = optim.AdamW([ 
        {'params': model.backbone.parameters(), 'lr': learning_rates["backbone"]},
        {'params': model.transformer.parameters(), 'lr': learning_rates["transformer"]},
        {'params': model.bert.parameters(), 'lr': learning_rates["bert"]},
        {'params': model.feat_map.parameters(), 'lr': learning_rates["feat_map"]},
        {'params': model.input_proj.parameters(), 'lr': learning_rates["input_proj"]},
        {'params': model.bbox_embed.parameters(), 'lr': learning_rates["bbox_embed"]},
        {'params': model.class_embed.parameters(), 'lr': learning_rates["class_embed"]}
    ], weight_decay=1e-4)  # weight_decay 

    # Cosine Annealing Scheduler 
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # freeze layers 
    layers_to_freeze = config['train']['layers_to_freeze']
    model = freeze_layers(model, layers_to_freeze)

    # Ensure the model is in training mode
    model.train()

    for epoch in range(epochs):
        total_loss = 0  # Track the total loss for this epoch
        for idx, (IMAGE_PATH, vals) in enumerate(ann_Dict.items()):
            image_source, image = load_image(IMAGE_PATH)
            bxs = vals['boxes']
            captions = vals['captions']

            # Zero the gradients
            optimizer.zero_grad()
            
            # Call the training function for each image and its annotations
            loss = train_image(
                model=model,
                image_source=image_source,
                image=image,
                caption_objects=captions,
                box_target=bxs,
            )
            
            # Backpropagate and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()  # Accumulate the loss
            print(f"Processed image {idx+1}/{len(ann_Dict)}, Loss: {loss.item()}")

        # scheduler step  
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Learning Rate: {scheduler.get_last_lr()}")

        # Print the average loss for the epoch
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(ann_Dict)}")

        if (epoch%save_epoch)==0:
            # Save the model's weights after each epoch
            torch.save(model.state_dict(), f"{save_path}{epoch}.pth")
            print(f"Model weights saved to {save_path}{epoch}.pth")



if __name__=="__main__":
    train(
        model=model, 
        ann_file=ann_file, 
        epochs=config['train']['num_epochs'], 
        learning_rate=config['train']['learning_rates'],
        save_path=config['train']['save_dir'],
        save_epoch=config['train']['save_epoch'],
    )
