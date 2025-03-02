## groundingDINO finetune

- 2023 net challenge AI code 
- groudningDINO finetune

[reference] 
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/Asad-Ismail/Grounding-Dino-FineTuning
  (from commit 57a66751ef68de19b3e9605a12f09a864c3e139a0)

[feature] 
- train config
- layer freeze
- learning rate scheduling
- layer-wise learning rate
- .. 

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


