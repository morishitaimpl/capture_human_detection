import sys, os, cv2, pathlib, shutil
sys.dont_write_bytecode = True
import numpy as np
import mediapipe as mp

images_dir = pathlib.Path(sys.argv[1])
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
img_paths = sorted([p for p in images_dir.iterdir() if p.suffix in IMG_EXTS])
output_dir = pathlib.Path(sys.argv[2]) # コピー先のディレクトリ
if(not output_dir.exists()): output_dir.mkdir() # ディレクトリ生成

mp_pose = mp.solutions.pose #姿勢推定
pose = mp_pose.Pose(static_image_mode=True)

for i in range(len(img_paths)):
    img = cv2.imread(img_paths[i])
    # print(img_paths[i].name)
    if img is not None:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_img)
    else:
        continue
    
    if results.pose_landmarks:
        print(f"Pose detected in {img_paths[i].name}")
        shutil.copy(str(img_paths[i]), str(output_dir))
    else:
        print(f"No pose detected in {img_paths[i].name}")
