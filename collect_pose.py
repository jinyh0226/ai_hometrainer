import time
import psycopg2
import cv2
import mediapipe as mp
import redis
import json

# PostgreSQL 연결 정보
DB_NAME = "pose_db"
DB_USER = "postgres"
DB_PASSWORD = "1234"
DB_HOST = "localhost"
DB_PORT = "5432"

# PostgreSQL 연결
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
    host=DB_HOST, port=DB_PORT
)
cur = conn.cursor()

# Redis 연결
r = redis.Redis(host='localhost', port=6379, db=0)

# Mediapipe 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 카메라
cap = cv2.VideoCapture(0)

last_saved_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        current_time = time.time()

        if current_time - last_saved_time >= 5.0:
            last_saved_time = current_time

            latest_landmarks = []

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                x, y, z = landmark.x, landmark.y, landmark.z
                latest_landmarks.append({
                    'id': idx,
                    'x': x,
                    'y': y,
                    'z': z
                })

                # PostgreSQL 저장
                cur.execute(
                    "INSERT INTO pose_landmarks (landmark_id, x, y, z) VALUES (%s, %s, %s, %s)",
                    (idx, x, y, z)
                )

            # Redis에 캐시 저장 (JSON)
            r.set('latest_pose', json.dumps(latest_landmarks))
            conn.commit()

            print("5초마다 DB 저장 + Redis 캐시")

        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Mediapipe Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()
cur.close()
conn.close()