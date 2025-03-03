from groundingdino.util.train import load_model, load_image, train_image, annotate
import torch

from train_settings import MODEL_CONFIG_PATH  #, MODEL_WEIGHT_PATH 

# reset test model weight path (optional)
MODEL_WEIGHT_PATH = "weights/knife_various2893_freezingNbatch50_test_1.pth"
# save path settings 
MODEL_SAVEPATH = "model.txt"
CHECKPOINT_SAVEPATH = "checkpoint_knife_various2893_freezingNbatch50_test_1.txt"

# Model ==========================================================
model = load_model(MODEL_CONFIG_PATH, MODEL_WEIGHT_PATH)
#print(model) 

# Model 구조 텍스트 파일로 저장
with open(MODEL_SAVEPATH, "w") as f:
    f.write(str(model))
    print(f"Saved {MODEL_SAVEPATH}!")

# Model Summary ==================================================
# import torch
# from torchsummary import summary

# def capture_summary(model, input_size): # summary 함수의 출력을 캡처 
#     from io import StringIO
#     import sys

#     old_stdout = sys.stdout # 임시로 출력 스트림을 변경하여 결과를 캡처
#     sys.stdout = buffer = StringIO()
#     summary(model, input_size=input_size) # 모델 요약 생성
#     sys.stdout = old_stdout # 출력 스트림을 원래대로 복원
#     return buffer.getvalue()

# (C, H, W) = (3, 600, 800)
# model_summary = capture_summary(model, (C, H, W))  

# with open("model_summary.txt", "w") as file:
#     file.write(model_summary) 

# Model checkpoint (weight값)=====================================
checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location="cpu")
#print(checkpoint) 

# Model weight값 텍스트 파일로 저장 
with open(CHECKPOINT_SAVEPATH, "w") as f:
    for key, value in checkpoint.items():
        # 텐서 데이터는 문자열 형태로 변환
        if isinstance(value, torch.Tensor):
            value_str = str(value.numpy())
        else:
            value_str = str(value)
        f.write(f"{key}: {value_str}\n\n")
    print(f"Saved {CHECKPOINT_SAVEPATH}!")