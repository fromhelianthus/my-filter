import cv2
import numpy as np

def main():
    # 얼굴 인식에 사용할 Haar Cascade 파일 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 웹캠에서 비디오 캡처
    cap = cv2.VideoCapture(0)

    # 오버레이할 이미지 로드
    filter_img = cv2.imread('pikachu.png', -1)

    # 필터 이미지가 제대로 로드되었는지 확인
    if filter_img is None:
        print("pikachu.png 파일을 찾을 수 없거나, 이미지가 잘못되었습니다.")
        return  # 필터 이미지가 없으면 프로그램 종료

    # 필터 이미지 크기 조정 비율 (예: 1.2배 확대)
    scale_factor = 1.2  # 필터 이미지를 1.2배 크게 만들기

    # 필터 이미지를 왼쪽으로 이동시키기 위한 값 (좌표 이동량, 예: 0.2 * w * scale_factor)
    x_shift = 0.3  # 필터 이미지를 20% 왼쪽으로 이동

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 탐지
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # 필터 이미지 크기 조정
            resized_filter = cv2.resize(filter_img, (int(w * scale_factor), int(h * scale_factor)))

            # 필터 적용 위치 계산 (y 값을 줄여서 위로 이동, y1과 y2 계산 시 크기 조정 비율을 적용)
            x1, x2 = x, x + int(w * scale_factor)
            y1, y2 = y - int(0.8 * h * scale_factor), y - int(0.8 * h * scale_factor) + int(h * scale_factor)  # y1을 조정하여 필터를 위로 이동

            # 필터 이미지의 알파 채널 분리
            alpha_s = resized_filter[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            # 필터를 얼굴에 오버레이
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * resized_filter[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])

        # 결과 출력
        cv2.imshow('Face Filter', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 캡처 해제 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
