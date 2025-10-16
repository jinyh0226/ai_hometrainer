import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import time
import pyttsx3
import threading
from queue import Queue
from collections import deque, Counter
from PIL import ImageFont, ImageDraw, Image

# --- TTS(음성) 설정 ---
tts_queue = Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 180) 
    while True:
        text_to_speak = tts_queue.get()
        if text_to_speak is None: break
        try:
            engine.say(text_to_speak)
            engine.runAndWait()
        except Exception: pass
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# --- 1. 기본 설정 및 함수 정의 ---
FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
try:
    font = ImageFont.truetype(FONT_PATH, 30)
except IOError:
    font = ImageFont.load_default()

def put_korean_text(img, text, position, font, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

# --- 2. 모델 로딩 및 운동 선택 ---
# LSTM 모델은 모든 운동을 판단하므로 하나만 로드
try:
    lstm_model = load_model("action_recognition_model.h5")
    encoder = joblib.load("action_encoder.joblib")
    print("LSTM 통합 모델을 성공적으로 로드했습니다.")
except Exception as e:
    print(f"오류: LSTM 모델 또는 인코더 파일 로드 실패. 스크립트를 종료합니다. {e}")
    tts_queue.put(None); tts_thread.join(); exit()

EXERCISE_LIST = ['pushup', 'dips', 'pullup', 'squat', 'plank']
while True:
    try:
        print("\n어떤 운동을 테스트하시겠습니까?")
        for i, name in enumerate(EXERCISE_LIST): print(f"{i+1}. {name}")
        choice = int(input("번호를 선택하세요: ")) - 1
        if 0 <= choice < len(EXERCISE_LIST):
            EXERCISE_NAME = EXERCISE_LIST[choice]
            break
        else: print("잘못된 번호입니다.")
    except ValueError: print("숫자만 입력해주세요.")

# --- 3. 변수 초기화 ---
current_state = {
    "stage": "down", 
    "reps": 0, "mistake_log": {},
    "start_time": 0, "total_hold_time": 0, "is_timing": False,
    "mistake_logged_this_rep": False 
}
last_mistake_feedback = "" # 마지막 피드백 내용 기억
last_feedback_time = 0     # 마지막 피드백 시간 기억
SAME_MISTAKE_COOLDOWN = 1.0 # 같은 실수에 대한 쿨다운
total_frames, correct_frames = 0, 0
last_elbow_angle = 0 

# LSTM 관련 변수
SEQUENCE_LENGTH = 30
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
lstm_prediction = ""

# --- 4. 웹캠 실행 및 메인 루프 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다."); tts_queue.put(None); tts_thread.join(); exit()

print("\n웹캠을 시작합니다. 'q' 키를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        if results.pose_landmarks:
            total_frames += 1
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            sequence_buffer.append(keypoints)
            
            landmarks_dict = {name: [landmarks[i].x, landmarks[i].y] for i, name in enumerate(mp_pose.PoseLandmark)}

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                input_seq = np.expand_dims(np.array(sequence_buffer), axis=0)
                prediction = lstm_model.predict(input_seq, verbose=0)[0]
                pred_index = np.argmax(prediction)
                lstm_prediction = encoder.inverse_transform([pred_index])[0]

            detailed_feedback = ""
            current_time = time.time()
            
            # --- 운동별 로직 ---
            # 1. 푸쉬업 (피드백 로직 수정)
            if EXERCISE_NAME == 'pushup':
                elbow_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_dict[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks_dict[mp_pose.PoseLandmark.LEFT_WRIST])
                back_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_dict[mp_pose.PoseLandmark.LEFT_HIP], landmarks_dict[mp_pose.PoseLandmark.LEFT_ANKLE])

                if elbow_angle > 160:
                    if current_state['stage'] == 'down': 
                        current_state['reps'] += 1
                        current_state['mistake_logged_this_rep'] = False
                    current_state['stage'] = 'up'
                elif elbow_angle < 90 and current_state['stage'] == 'up':
                    current_state['stage'] = 'down'
                
                if back_angle < 165 or back_angle > 195:
                    detailed_feedback = "허리를 펴세요!"
                elif current_state['stage'] == 'up' and 90 < elbow_angle <= 160:
                    detailed_feedback = "더 깊게 내려가세요!"
                elif lstm_prediction == 'pushup_wrong': 
                    detailed_feedback = "자세가 불안정합니다."

            # 2. 딥스
            elif EXERCISE_NAME == 'dips':
                elbow_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_dict[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks_dict[mp_pose.PoseLandmark.LEFT_WRIST])
                if elbow_angle > 160:
                    if current_state['stage'] == 'down': 
                        current_state['reps'] += 1
                        current_state['mistake_logged_this_rep'] = False
                    current_state['stage'] = 'up'
                elif elbow_angle < 90 and current_state['stage'] == 'up':
                    current_state['stage'] = 'down'

                if lstm_prediction == 'dips_wrong': detailed_feedback = "어깨를 뒤로 고정하세요!"
                elif current_state['stage'] == 'up' and 90 < elbow_angle <= 160: detailed_feedback = "더 깊게 내려가세요!"

            # 3. 풀업 (피드백 로직 수정)
            elif EXERCISE_NAME == 'pullup':
                elbow_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_dict[mp_pose.PoseLandmark.LEFT_ELBOW], landmarks_dict[mp_pose.PoseLandmark.LEFT_WRIST])
                is_going_up = elbow_angle < last_elbow_angle
                last_elbow_angle = elbow_angle

                if current_state['stage'] == 'down' and elbow_angle < 90:
                    current_state['stage'] = 'up'
                elif current_state['stage'] == 'up' and elbow_angle > 160:
                    current_state['reps'] += 1
                    current_state['mistake_logged_this_rep'] = False
                    current_state['stage'] = 'down'
                
                if lstm_prediction == 'pullup_wrong': detailed_feedback = "반동을 사용하지 마세요."
                # [수정] 동작의 중간 구간(90~160도)에서 올라가는 중일 때만 피드백
                elif current_state['stage'] == 'down' and is_going_up and (90 < elbow_angle <= 160):
                    detailed_feedback = "더 높이 올라가세요!"

            # 4. 스쿼트
            elif EXERCISE_NAME == 'squat':
                knee_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_HIP], landmarks_dict[mp_pose.PoseLandmark.LEFT_KNEE], landmarks_dict[mp_pose.PoseLandmark.LEFT_ANKLE])
                back_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_dict[mp_pose.PoseLandmark.LEFT_HIP], landmarks_dict[mp_pose.PoseLandmark.LEFT_KNEE])
                
                if knee_angle > 160:
                    if current_state['stage'] == 'down': 
                        current_state['reps'] += 1
                        current_state['mistake_logged_this_rep'] = False
                    current_state['stage'] = 'up'
                elif knee_angle < 90 and current_state['stage'] == 'up':
                    current_state['stage'] = 'down'

                if knee_angle < 150 and back_angle < 75: detailed_feedback = "허리를 펴세요!"
                elif current_state['stage'] == 'up' and 90 < knee_angle <= 160: detailed_feedback = "더 깊게 내려가세요!"
                elif lstm_prediction == 'squat_wrong': detailed_feedback = "자세가 불안정합니다."

            # 5. 플랭크
            elif EXERCISE_NAME == 'plank':
                is_correct_pose = (lstm_prediction == 'plank_correct')
                if is_correct_pose and not current_state['is_timing']:
                    current_state['start_time'] = current_time
                    current_state['is_timing'] = True
                elif not is_correct_pose and current_state['is_timing']:
                    hold_duration = current_time - current_state['start_time']
                    current_state['total_hold_time'] += hold_duration
                    current_state['is_timing'] = False
                
                if not is_correct_pose:
                    back_angle = calculate_angle(landmarks_dict[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks_dict[mp_pose.PoseLandmark.LEFT_HIP], landmarks_dict[mp_pose.PoseLandmark.LEFT_ANKLE])
                    if back_angle <= 170: detailed_feedback = "엉덩이를 조금만 더 낮추세요!"
                    elif back_angle >= 190: detailed_feedback = "허리가 처지지 않게 배에 힘을 주세요!"
            
            # --- 피드백 기록 및 화면 표시 ---
            if not detailed_feedback:
                correct_frames += 1
            else:
                speak_now = False
                if tts_queue.empty():
                    if detailed_feedback != last_mistake_feedback:
                        speak_now = True
                    elif current_time - last_feedback_time > SAME_MISTAKE_COOLDOWN:
                        speak_now = True
                
                if speak_now:
                    tts_queue.put(detailed_feedback)
                    last_mistake_feedback = detailed_feedback
                    last_feedback_time = current_time
                
                if not current_state['mistake_logged_this_rep']:
                    current_state["mistake_log"][detailed_feedback] = current_state["mistake_log"].get(detailed_feedback, 0) + 1
                    current_state['mistake_logged_this_rep'] = True

            y_pos = 10
            image = put_korean_text(image, f"운동: {EXERCISE_NAME.capitalize()}", (10, y_pos), font, (255, 191, 0)); y_pos += 50
            
            if EXERCISE_NAME == 'plank':
                display_time = current_state['total_hold_time']
                if current_state['is_timing']: display_time += time.time() - current_state['start_time']
                image = put_korean_text(image, f"시간: {display_time:.1f} 초", (10, y_pos), font, (0, 255, 0)); y_pos += 50
            else:
                image = put_korean_text(image, f"횟수: {current_state['reps']}", (10, y_pos), font, (0, 255, 0)); y_pos += 50
            
            feedback_text = detailed_feedback if detailed_feedback else "자세가 좋습니다!"
            color = (0, 0, 255) if detailed_feedback else (0, 255, 0)
            image = put_korean_text(image, f"피드백: {feedback_text}", (10, y_pos), font, color); y_pos += 50
            
            lstm_text = f"움직임: {lstm_prediction}" if lstm_prediction else "움직임: 분석 중..."
            image = put_korean_text(image, lstm_text, (10, y_pos), font, (255, 255, 255))

    except Exception as e: pass

    cv2.imshow('AI Fitness Trainer - LSTM Integrated', image)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

# --- 5. 종료 및 최종 리포트 ---
print("\n운동을 종료합니다."); tts_queue.put("운동을 종료합니다.")
cap.release(); cv2.destroyAllWindows()

print("\n--- 최종 운동 리포트 ---")
if EXERCISE_NAME != 'plank':
    print(f"총 횟수: {current_state['reps']}회")
    tts_queue.put(f"총 {current_state['reps']}회 완료했습니다.")
else:
    final_time = current_state['total_hold_time']
    if current_state['is_timing']: final_time += time.time() - current_state['start_time']
    print(f"총 유지 시간: {final_time:.1f}초")
    tts_queue.put(f"총 {int(final_time)}초 동안 유지했습니다.")

if total_frames > 0:
    accuracy = (correct_frames / total_frames) * 100
    print(f"평균 자세 정확도: {accuracy:.2f}%")
    tts_queue.put(f"평균 자세 정확도는 약 {int(accuracy)}퍼센트입니다.")
else:
    print("운동 데이터가 부족하여 정확도를 계산할 수 없습니다.")

if current_state['mistake_log']:
    print("\n--- AI 코치 조언 ---")
    tts_queue.put("AI 코치 조언입니다.")
    
    # [수정] "팔을 완전히 펴세요" 조언 제거
    improvement_tips = {
        "허리를 펴세요!": "가장 많이 한 실수는 허리를 편 자세를 유지하지 못한 것입니다. 다음에는 복부에 힘을 주고, 몸을 일직선으로 만드는데 집중해보세요.",
        "더 깊게 내려가세요!": "가장 많이 한 실수는 충분한 가동범위를 활용하지 못한 것입니다. 다음에는 조금 더 깊게 내려가 근육의 이완을 최대로 이끌어내는 데 집중해보세요.",
        "어깨를 뒤로 고정하세요!": "가장 많이 한 실수는 어깨가 앞으로 말리는 것이었습니다. 다음에는 가슴을 펴고 어깨를 뒤로 고정시킨 상태를 유지하는데 집중해보세요.",
        "반동을 사용하지 마세요.": "가장 많이 한 실수는 반동을 사용하는 것이었습니다. 다음에는 몸의 흔들림을 최소화하고 등의 힘으로 몸을 끌어올리는 데 집중해보세요.",
        "더 높이 올라가세요!": "가장 많이 한 실수는 충분히 높이 올라가지 못한 것입니다. 다음에는 턱이 봉 위로 올라갈 때까지 몸을 당기는 데 집중해보세요.",
        "자세가 불안정합니다.": "전체적으로 자세가 불안정한 경우가 많았습니다. 다음에는 발바닥 전체로 땅을 단단히 누르며 안정적인 기반을 만드는 데 집중해보세요.",
        "엉덩이를 조금만 더 낮추세요!": "가장 많이 한 실수는 엉덩이가 높이 들리는 것이었습니다. 다음에는 몸이 일직선이 되도록 엉덩이를 조금 더 낮추는 데 집중해보세요.",
        "허리가 처지지 않게 배에 힘을 주세요!": "가장 많이 한 실수는 허리가 아래로 처지는 것이었습니다. 다음에는 배에 힘을 꽉 주어 허리가 아치형이 되지 않도록 유지하는 데 집중해보세요."
    }

    most_common_mistake = max(current_state['mistake_log'], key=current_state['mistake_log'].get)
    report_message = improvement_tips.get(most_common_mistake, f"가장 자주 틀린 자세는 '{most_common_mistake}' 입니다. 다음에는 이 점에 더 신경 써보세요!")

    print(report_message)
    tts_queue.put(report_message)

else:
    success_message = "모든 자세를 완벽하게 수행했습니다. 정말 대단해요!"
    print(success_message)
    tts_queue.put(success_message)

tts_queue.put(None); tts_thread.join()
print("\n프로그램을 완전히 종료합니다.")
