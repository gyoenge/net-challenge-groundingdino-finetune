'''
    vis_dataset_save.py 
    : training에 사용할 image dataset에 대해, bbox annotation이 모두 올바르게 되어있는지 확인하는 코드 
    (RESULT_PATH 폴더에 vis_dataset 결과 save)
'''

import cv2 
import os
import csv
from collections import defaultdict

from train_settings import IMAGES_DIR, ANN_FILE
RESULT_PATH = "vis_dataset/"

def load_annotation(ann_file):
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

def draw_bbox_withlabel(img, box, caption):
    x1, y1, x2, y2 = box

    color = (0, 0, 255)  # 빨간색
    thickness = 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    label_position = (x1, y1-10)
    font_scale=0.5
    cv2.putText(img, caption, label_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

    return img 

def process_image(idx, img, bxs, captions):
    for bbox, caption in zip(bxs, captions):
        img = draw_bbox_withlabel(img, bbox, caption)
    return img 

if __name__=="__main__":
    # load dataset 
    images_files=sorted(os.listdir(IMAGES_DIR)) 
    ann_Dict = load_annotation(ANN_FILE) 
    
    # save after drawing bbox on image dataset 
    for idx, (IMAGE_PATH, vals) in enumerate(ann_Dict.items()):
        image = cv2.imread(IMAGE_PATH)
        bxs = vals['boxes']
        captions = vals['captions']
        drawn = process_image(idx, image, bxs, captions)
        
        img_name = os.path.basename(IMAGE_PATH)
        result_file = RESULT_PATH + f"VIS_{img_name}"
        cv2.imwrite(result_file, drawn)
        