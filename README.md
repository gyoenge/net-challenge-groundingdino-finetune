## groundingDINO finetune

- 2023 net challenge AI code 
- groudningDINO finetune

[reference] 
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/Asad-Ismail/Grounding-Dino-FineTuning
  (from commit 57a66751ef68de19b3e9605a12f09a864c3e139a0)

[feature] 
- (train, eval) yaml config 
- (train) freeze layers 
- (train) learning rate scheduling 
- (train) layer-wise learning rate
- (eval) speed test 
- (eval) precision, recall, AP50, AP50-95 
- (eval) show confusion matrix 
- (eval) save visualization of predicted box & ground truth 
- ... 

---

### Installation

- check CUDA : read IDEA-Research/GroundingDINO/README.md
- install requirements
  ```
  cd GroundingDINO/
  pip install -e .
  ```
- install weights
  ```
  mkdir weights
  cd weights
  wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
  cd ..
  ```


### train 

- change train config : change train_config.yaml 
- run train 
  ```
  python train.py
  ```


### evaluation 

- change evaluation config : change evaluation_config.yaml
- ** dataset should be matched with multimodal-data (see uploaded example)
- run evaluation 
  ```
  python evaluation.py
  ```
