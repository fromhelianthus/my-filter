from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# 얼굴 인식에 사용할 Haar Cascade 파일 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 오버레이할 이미지 로드
filter_img = cv2.imread('pikachu.png', -1)

# 필터 이미지 크기 조정 비율 (예: 1.2배 확대)
scale_factor = 1.2  # 필터 이미지를 1.2배 크게 만들기

# 필터 이미지를 왼쪽으로 이동시키기 위한 값 (좌표 이동량, 예: 0.2 * w * scale_factor)
x_shift = 0.05  # 필터 이미지를 5% 왼쪽으로 이동

def apply_filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        resized_filter = cv2.resize(filter_img, (int(w * scale_factor), int(h * scale_factor)))
        x1, x2 = x - int(x_shift * w * scale_factor), x - int(x_shift * w * scale_factor) + int(w * scale_factor)
        y1, y2 = y - int(0.7 * h * scale_factor), y - int(0.7 * h * scale_factor) + int(h * scale_factor)

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > frame.shape[1]:
            x2 = frame.shape[1]
        if y2 > frame.shape[0]:
            y2 = frame.shape[0]

        alpha_s = resized_filter[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        filter_y1, filter_y2 = max(0, y1), min(frame.shape[0], y2)
        filter_x1, filter_x2 = max(0, x1), min(frame.shape[1], x2)

        frame[filter_y1:filter_y2, filter_x1:filter_x2, :] = (
            alpha_s[(filter_y1 - y1):(filter_y2 - y1), (filter_x1 - x1):(filter_x2 - x1)] * resized_filter[
                                                                                  (filter_y1 - y1):(filter_y2 - y1),
                                                                                  (filter_x1 - x1):(filter_x2 - x1), :3] +
            alpha_l[(filter_y1 - y1):(filter_y2 - y1), (filter_x1 - x1):(filter_x2 - x1)] * frame[
                                                                                          filter_y1:filter_y2,
                                                                                          filter_x1:filter_x2, :3]
        )
    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # FPS 설정
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = apply_filter(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
