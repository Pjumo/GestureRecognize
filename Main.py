import cv2
import mediapipe as mp

# 손가락 마디를 그릴 수 있게 하는 유틸
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 카메라 웹캠 열기 (캠이 여러 개인 경우 숫자 신경 써서)
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        max_num_hands=1,  # 손 인식 개수
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()  # 카메라 한 프레임 읽어 오기
        if not success:
            continue

        # OpenCV는 BGR, Mediapipe는 RGB를 사용하기 때문에 형식 변환
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)  # 이미지 전처리

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('image', image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
