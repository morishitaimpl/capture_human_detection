import sys, os, time
sys.dont_write_bytecode = True
import torch
from torch import nn
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pathlib
import cv2
import config as cf
import load_dataset_annot as ld

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def load_model(model_path=None):
    """モデルの読み込み"""
    model = cf.build_model()
    if model_path and os.path.exists(model_path):
        if DEVICE == "cuda":
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path, torch.device("cpu")))
        print(f"Model loaded from: {model_path}")
    else:
        print("Warning: No model file provided or file not found. Using untrained model.")
    
    model.to(DEVICE)
    model.eval()
    return model

def detect_humans_realtime(model_path=None):
    """リアルタイム人検出"""
    model = load_model(model_path)
    
    data_transforms = T.Compose([T.ToTensor()])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: カメラを開けませんでした")
        return
    
    print("カメラが起動しました。'q'キーで終了します。")
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: フレームを読み込めませんでした")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            data = data_transforms(pil_image)
            data = data.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = model(data)
            
            boxes = outputs[0]["boxes"].detach().cpu().numpy()
            scores = outputs[0]["scores"].detach().cpu().numpy()
            labels = outputs[0]["labels"].detach().cpu().numpy()
            
            human_detected = False
            
            for i in range(len(scores)):
                if scores[i] < cf.thDetection:
                    continue
                
                if labels[i] == 1:  # 人のクラス
                    human_detected = True
                    
                    x1, y1, x2, y2 = boxes[i].astype(int)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    label_text = f"Person: {scores[i]:.3f}"
                    label_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                    
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), (0, 0, 255), -1)
                    
                    cv2.putText(frame, label_text, (x1, y1 - 5), 
                              font, font_scale, (255, 255, 255), thickness)
            
            if human_detected:
                warning_text = "人がいます。注意してください。"
                text_size = cv2.getTextSize(warning_text, font, 1.0, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 40
                
                cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                            (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
                
                cv2.putText(frame, warning_text, (text_x, text_y), 
                          font, 1.0, (255, 255, 255), 2)
            
            cv2.imshow('Human Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n終了しています...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    detect_humans_realtime(model_path)

