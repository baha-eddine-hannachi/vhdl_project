import cv2
from ultralytics import YOLO
import torch
import time
from collections import deque

print("Chargement du modèle...")
try:
    model = YOLO('yolov8n.pt') 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model.to('cuda')
        model.model.half()  # Convert model to half precision
        print(f"Modèle chargé sur {device} (mode FP16)")
    else:
        print(f"Modèle chargé sur {device}")
except Exception as e:
    print(f"Erreur chargement modèle: {e}")
    exit(1)

# --- CONFIGURATION CAMERA ---
camera_index = 0
patience_frames = 30
absence_counter = 0
presence_status = False
num_persons = 0  
detection_actuelle = False 
#n7oto fiha les coordonnées des boîtes détectées pour les dessiner à chaque frame, même si la détection n'est pas effectuée à chaque frame
detected_boxes = []  # List to store (x1, y1, x2, y2, confidence) tuples

# FPS calculation variables
fps_history = deque(maxlen=30)
prev_time = time.time()

# Open camera 
print(f"Ouverture caméra {camera_index}...")
cap = cv2.VideoCapture(camera_index)

# Try different resolutions: 640x480 is usually good balance
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

if not cap.isOpened():
    print(f"Erreur: Impossible d'ouvrir la caméra {camera_index}")
    exit(1)

print("Démarrage de la détection... (Appuyez sur 'q' pour quitter)")
print(f"Résolution: {desired_width}x{desired_height}")

frame_count = 0
process_every_n_frames = 2  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lecture frame")
        break
    
    frame_count += 1
    
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)
    if frame_count % process_every_n_frames == 0:
        display_frame = frame.copy()        
        try:
            inference_frame = cv2.resize(frame, (320, 240))
            
            if device == 'cuda':
                results = model(inference_frame, classes=[0], conf=0.5, verbose=False, 
                              imgsz=320, half=True)  # half precision
            else:
                results = model(inference_frame, classes=[0], conf=0.5, verbose=False)
            
            num_persons = 0
            detection_actuelle = False
            detected_boxes = [] 
            
            if len(results) > 0 and results[0].boxes is not None:
                num_persons = len(results[0].boxes)
                detection_actuelle = num_persons > 0

                h, w = frame.shape[:2]
                scale_x = w / 320
                scale_y = h / 240
                
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    # Scale coordinates
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Store box coordinates for drawing in every frame
                    detected_boxes.append((x1, y1, x2, y2, confidence))

        except Exception as e:
            print(f"Erreur IA: {e}")
        
    
    for (x1, y1, x2, y2, confidence) in detected_boxes:
        # Draw green rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with confidence
        label = f"Personne {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if detection_actuelle:
        absence_counter = 0
        presence_status = True
        status_text = f"PRESENT: {num_persons} personne(s)"
        status_color = (0, 255, 0)
    else:
        if presence_status:
            absence_counter += 1
            if absence_counter > patience_frames:
                presence_status = False
                absence_counter = 0
            status_text = "Cible perdue..."
            status_color = (0, 165, 255)
        else:
            status_text = "ZONE VIDE"
            status_color = (0, 0, 255)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), status_color, -1)
    cv2.putText(frame, status_text, (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display FPS and settings
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Skip: {process_every_n_frames-1}", (150, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Display number of detected persons
    cv2.putText(frame, f"Detections: {len(detected_boxes)}", (300, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Detection YOLO - Personnes', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Programme terminé")