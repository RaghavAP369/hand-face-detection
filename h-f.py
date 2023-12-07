import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def is_hand_near_face(hand_landmarks, face_box):
    if hand_landmarks is not None and face_box is not None:
        hand_x, hand_y = hand_landmarks[8]['x'], hand_landmarks[8]['y']
        face_x, face_y, face_w, face_h = face_box

        distance_threshold = 50
        if abs(hand_x - face_x) < distance_threshold and abs(hand_y - face_y) < distance_threshold:
            return True

    return False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = hands.process(rgb_frame)

    
    hand_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

    
    face_box = (100, 100, 200, 200)

   
    if is_hand_near_face(hand_landmarks, face_box):
        cv2.putText(frame, 'Hand near face!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

   
    if hand_landmarks:
        mp.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[0] + face_box[2], face_box[1] + face_box[3]), (0, 255, 0), 2)

    cv2.imshow('Hand and Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
