import cv2
import mediapipe as mp
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import time
import sys



# Image transform for model
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# 5초 이상 눈이 감긴 경우 추적을 위한 변수 설정
left_eye_closed_start = None
right_eye_closed_start = None
warning_displayed = False  # 경고 메시지가 표시되었는지 추적



# PyTorch 모델 정의 및 로드
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("checkpoint_epoch_2.pth.tar")['state_dict'])
model.eval()


# MediaPipe 설정
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


# 랜드마크 인덱스 (눈 영역)
LEFT_EYE_INDEXES = [33, 133, 160, 144, 145, 153, 154, 155]
RIGHT_EYE_INDEXES = [362, 263, 387, 373, 374, 380, 381, 382]

# Mesh 변수 설정
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

video = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, image = video.read()
        if not ret:
            break
        
        # BGR -> RGB 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, c = image.shape  # 이미지의 높이, 너비 가져오기
                
                # 눈 영역 좌표 계산
                left_eye = [(int(face_landmarks.landmark[i].x * w), 
                             int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_INDEXES]
                right_eye = [(int(face_landmarks.landmark[i].x * w), 
                              int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_INDEXES]

                # 눈 영역 잘라내기 (OpenCV를 이용해 사각형 영역 추출)
                left_x, left_y, left_w, left_h = cv2.boundingRect(np.array(left_eye))
                right_x, right_y, right_w, right_h = cv2.boundingRect(np.array(right_eye))

                # 패딩 설정
                padding_x = 30  # 좌우 방향 패딩
                padding_y_top = 80  # 위쪽 패딩 (눈썹 포함)
                padding_y_bottom = 20  # 아래쪽 패딩 (상대적으로 적게)

                # 왼쪽 눈 좌표에 패딩 적용
                left_x = max(0, left_x - padding_x)
                left_y = max(0, left_y - padding_y_top)  # 위쪽으로 더 많은 여유 추가
                left_w += 2 * padding_x
                left_h += padding_y_top + padding_y_bottom

                # 오른쪽 눈 좌표에 패딩 적용
                right_x = max(0, right_x - padding_x)
                right_y = max(0, right_y - padding_y_top)
                right_w += 2 * padding_x
                right_h += padding_y_top + padding_y_bottom

                left_eye_img = image[left_y:left_y+left_h, left_x:left_x+left_w]
                right_eye_img = image[right_y:right_y+right_h, right_x:right_x+right_w]

                left_eye_img = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2RGB)
                right_eye_img = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2RGB)


                # 디버깅용 사진 저장
                # cv2.imwrite("right_eye_debug.jpg", right_eye_img)
                # cv2.imwrite("left_eye_debug.jpg", left_eye_img)

                left_eye_transformed_image = my_transforms(left_eye_img).unsqueeze(0)
                right_eye_transformed_image = my_transforms(right_eye_img).unsqueeze(0)

                # ResNet 모델 예측
                left_eye_pred = model(left_eye_transformed_image)
                right_eye_pred = model(right_eye_transformed_image)

                # 결과 해석
                left_eye_status = "Open" if left_eye_pred[0][0] < 0.5 else "Closed"
                right_eye_status = "Open" if right_eye_pred[0][0] < 0.5 else "Closed"


                # 눈 감음 상태 추적
                current_time = time.time()
                # 왼쪽 눈 감음 상태 추적
                if left_eye_status == "Closed":
                    if left_eye_closed_start is None:  # 처음 감기 시작     시간 기록
                        left_eye_closed_start = current_time
                else:
                    left_eye_closed_start = None  # 눈을 뜨면 초기화

                # 오른쪽 눈 감음 상태 추적
                if right_eye_status == "Closed":
                    if right_eye_closed_start is None:  # 처음 감기 시작    시간 기록
                        right_eye_closed_start = current_time
                else:
                    right_eye_closed_start = None  # 눈을 뜨면 초기화

                # 경고 조건 확인 (둘 다 5초 이상 감긴 경우)
                if (left_eye_closed_start and right_eye_closed_start and
                    current_time - left_eye_closed_start >= 5 and
                    current_time - right_eye_closed_start >= 5):
                    warning_displayed = True
                    cv2.putText(image, "WARNING: Eyes closed for 5 seconds!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                # 결과 화면에 표시
                cv2.putText(image, f"Left Eye: {left_eye_status}", (left_x, left_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(image, f"Right Eye: {right_eye_status}", (right_x, right_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 눈 영역 디버깅용 사각형 그리기
                cv2.rectangle(image, (left_x, left_y), (left_x+left_w, left_y+left_h), (255, 0, 0), 1)
                cv2.rectangle(image, (right_x, right_y), (right_x+right_w, right_y+right_h), (255, 0, 0), 1)

        # 결과 이미지 출력
        cv2.imshow("Face Mesh", image)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()