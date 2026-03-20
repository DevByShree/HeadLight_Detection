import cv2
import os
import numpy as np

# This code is for only extracting the frames from videos

videos = [
    "Videos/V1.webm",
    "Videos/V2.webm",
    "Videos/V3.webm",
    "Videos/V4.webm",
    "Videos/V5.webm",
]
os.makedirs("all_frames",exist_ok=True)

total_saved=0

for video_path in videos:

    if not os.path.exists(video_path):
        print("Video file does not exist")
        continue

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]  

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 {video_name}: {total_frames} frames, {fps} FPS")
    
    count = 0
    saved = 0
    
    frame_gap = 10
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % frame_gap == 0:
            filename = f"{video_name}_f{count:05d}.jpg"
            filepath = os.path.join("all_frames", filename)
            cv2.imwrite(filepath, frame)
            saved += 1
        
        count += 1
    
    cap.release()
    total_saved += saved
    print(f"    {saved} frames saved")

print(f"\n{'='*40}")
print(f" Total frames saved: {total_saved}")
print(f" Saved in: all_frames/")
print(f"\nNext: run step2 - sort the frames")