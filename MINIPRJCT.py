import cv2
import mediapipe as mp
import pyautogui
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
PINCH_THRESHOLD = 0.05
ZOOM_THRESHOLD = 0.1
scroll_start_y = None

def is_pinching(landmarks, finger1, finger2, threshold=PINCH_THRESHOLD):
    thumb_landmark = landmarks.landmark[finger1]
    other_finger_landmark = landmarks.landmark[finger2]
    distance = np.linalg.norm(np.array([thumb_landmark.x, thumb_landmark.y]) - 
                              np.array([other_finger_landmark.x, other_finger_landmark.y]))
    return distance < threshold

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[INDEX_TIP]
        middle_finger_tip = hand_landmarks.landmark[MIDDLE_TIP]
        x = int(index_finger_tip.x * frame.shape[1])
        y = int(index_finger_tip.y * frame.shape[0])
        screen_x = np.clip(x, 0, screen_width - 1)
        screen_y = np.clip(y, 0, screen_height - 1)
        pyautogui.moveTo(screen_x, screen_y)
        if is_pinching(hand_landmarks, THUMB_TIP, INDEX_TIP):
            pyautogui.click(button='left')
            print("Left Click")
        
        if is_pinching(hand_landmarks, THUMB_TIP, MIDDLE_TIP):
            pyautogui.click(button='right')
            print("Right Click")
        if scroll_start_y is None:
            scroll_start_y = y
        else:
            scroll_direction = y - scroll_start_y
            if abs(scroll_direction) > 20:
                pyautogui.scroll(-scroll_direction)
                scroll_start_y = y
                print("Scrolling:", "Down" if scroll_direction > 0 else "Up")
        if is_pinching(hand_landmarks, INDEX_TIP, MIDDLE_TIP, threshold=ZOOM_THRESHOLD):
            zoom_distance = np.linalg.norm(np.array([index_finger_tip.x, index_finger_tip.y]) -
                                           np.array([middle_finger_tip.x, middle_finger_tip.y]))
            if zoom_distance < ZOOM_THRESHOLD / 2:
                pyautogui.hotkey("ctrl", "-")
                print("Zooming Out")
            else:
                pyautogui.hotkey("ctrl", "+")
                print("Zooming In")
        if index_finger_tip.x < 0.1:
            pyautogui.hotkey('alt', 'left')
            print("Switched Left")
        elif index_finger_tip.x > 0.9:
            pyautogui.hotkey('alt', 'right')
            print("Switched Right")

        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
