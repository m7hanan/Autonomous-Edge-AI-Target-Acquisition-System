import cv2
import numpy as np

# --- SYSTEM CONFIGURATION ---
DEADZONE_RADIUS = 15  # Anti-jitter filter radius in pixels

print("Loading Edge AI Model (ResNet-10 SSD)...")
# Ensure the .prototxt and .caffemodel files are in the same directory
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

cap = cv2.VideoCapture(0)
print("Autonomous Face-Tracking System Active. Press 'q' to exit.")

# Variables to store previous coordinates for the Deadzone logic
prev_target_x, prev_target_y = -1, -1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed lost!")
        break
        
    (h, w) = frame.shape[:2]
    center_x_frame = w // 2
    center_y_frame = h // 2
    
    # Preprocess frame for the Deep Learning neural network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections (50% confidence threshold)
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # --- 1. FACIAL LANDMARK OFFSET CORRECTION ---
            # X remains in the horizontal center
            raw_target_x = startX + (endX - startX) // 2
            # Y is shifted to the top 25% of the bounding box
            raw_target_y = startY + int((endY - startY) * 0.25)
            
            # --- 2. ANTI-JITTER DEADZONE FILTER ---
            if prev_target_x == -1:
                target_x, target_y = raw_target_x, raw_target_y
            else:
                # Calculate pixel distance between new movement and previous position
                distance = np.sqrt((raw_target_x - prev_target_x)**2 + (raw_target_y - prev_target_y)**2)
                
                if distance > DEADZONE_RADIUS:
                    target_x, target_y = raw_target_x, raw_target_y # Move to new position
                else:
                    target_x, target_y = prev_target_x, prev_target_y # Ignore micro-fluctuations (Jitter)
            
            # Update previous coordinates
            prev_target_x, prev_target_y = target_x, target_y
            
            # --- 3. TELEMETRY OUTPUT (For Phase 2 PID Controller) ---
            # Calculates how far the target is from the absolute center of the camera
            delta_x = target_x - center_x_frame
            delta_y = target_y - center_y_frame
            
            # --- 4. TRACKING RETICLE UI ---
            reticle_radius = 25
            color = (0, 255, 0) # Professional Green
            thickness = 2
            
            # Draw base bounding box (subtle)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (50, 50, 50), 1)
            
            # Draw Reticle Center Dot (Red)
            cv2.circle(frame, (target_x, target_y), 3, (0, 0, 255), -1)
            
            # Draw Outer Reticle Scope
            cv2.circle(frame, (target_x, target_y), reticle_radius, color, thickness)
            
            # Draw Scope Crosslines
            cv2.line(frame, (target_x - reticle_radius - 15, target_y), (target_x + reticle_radius + 15, target_y), color, thickness)
            cv2.line(frame, (target_x, target_y - reticle_radius - 15), (target_x, target_y + reticle_radius + 15), color, thickness)
            
            # HUD Text overlay
            text = f"TRACKING ACTIVE | CONFIDENCE: {confidence * 100:.1f}%"
            cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the final processed frame
    cv2.imshow('Autonomous Face-Tracking HUD', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
