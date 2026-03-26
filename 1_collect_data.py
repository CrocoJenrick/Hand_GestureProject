import cv2
import mediapipe as mp
import pandas as pd

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
except ImportError:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_draw

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

data = []
cap = cv2.VideoCapture(0)

print("Press 'l' for Like, 'p' for Peace, and 'f' for Fuck U. Press 'q' to quit and save.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            row = []
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('l'): 
                data.append(row + ["like"])
                print("Saved Like")
            elif key == ord('p'): 
                data.append(row + ["peace"])
                print("Saved Peace")
            elif key == ord('f'): 
                data.append(row + ["fuck_u"])
                print("Saved Fuck U")

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save to CSV
columns = []
for i in range(21): columns.extend([f'x{i}', f'y{i}', f'z{i}'])
columns.append('label')
df = pd.DataFrame(data, columns=columns)
df.to_csv('gesture_dataset.csv', index=False)
print("Dataset saved! You can now run the training script.")

cap.release()
cv2.destroyAllWindows()