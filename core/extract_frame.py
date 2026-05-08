import os
import cv2
import json
import argparse
from tqdm import tqdm
from PIL import Image
from mtcnn import MTCNN

detector = MTCNN()

def crop_and_resize_face(frame_rgb, margin=0.2, target_size=(448, 448)):
    faces = detector.detect_faces(frame_rgb)
    if not faces:
        return None
    
    main_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
    x, y, w, h = main_face['box']
    
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    img_h, img_w, _ = frame_rgb.shape
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    
    face_crop = frame_rgb[y1:y2, x1:x2]
    
    try:
        face_pil = Image.fromarray(face_crop)
        face_resized = face_pil.resize(target_size, Image.LANCZOS)
        return face_resized
    except Exception as e:
        return None

def process_video(video_path, output_dir, num_frames=5):
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        cap.release()
        return []

    interval = total_frames // num_frames
    saved_files = []
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i in range(num_frames):
        frame_idx = i * interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_img = crop_and_resize_face(frame_rgb)
            
            if face_img:
                out_filename = f"{base_name}_frame{frame_idx}.jpg"
                out_filepath = os.path.join(output_dir, out_filename)
                
                face_img.save(out_filepath, quality=95)
                saved_files.append(out_filename)

    cap.release()
    return saved_files

def main():
    parser = argparse.ArgumentParser(description="Extract and save face frames from videos for deepfake detection.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing original videos (.mp4)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save face images (.jpg)")
    parser.add_argument("--frames_per_video", type=int, default=5, help="Number of frames to extract per video")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_files = [f for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi'))]
    print(f"Found {len(video_files)} videos. Starting processing...")

    total_images_saved = 0
    
    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(args.video_dir, video_file)
        saved_imgs = process_video(video_path, args.output_dir, args.frames_per_video)
        total_images_saved += len(saved_imgs)

    print("\n" + "="*50)
    print("Extraction Completed!")
    print(f"Videos processed: {len(video_files)}")
    print(f"Total images saved: {total_images_saved}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    main()