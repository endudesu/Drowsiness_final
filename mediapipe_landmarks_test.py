"""
학습 환경 모니터링 시스템
- MediaPipe를 사용한 자세 감지
- PyQt5를 사용한 실시간 UI
- ROI 기반 다중 사용자 모니터링

작성자: [작성자명]
최종 수정일: [날짜]
"""

# 필요한 라이브러리 임포트
#-------------------------------------------
import cv2 as cv
import numpy as np
import mediapipe as mp
import time
from datetime import datetime
import os
import sys
from AngleBuffer import AngleBuffer

# PyQt 관련 임포트
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, 
    QVBoxLayout, QHBoxLayout, QFrame, QScrollArea,
    QPushButton, QSpinBox
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QColor, QFont,
    QBrush, QPen, QLinearGradient
)

#-------------------------------------------
# 상수 정의
#-------------------------------------------

# MediaPipe 모델 설정 관련 상수
MIN_DETECTION_CONFIDENCE = 0.5  # MediaPipe 모델의 최소 감지 신뢰도 (0.0 ~ 1.0)
MIN_TRACKING_CONFIDENCE = 0.5   # MediaPipe 모델의 최소 추적 신뢰도 (0.0 ~ 1.0)
BLINK_THRESHOLD = 0.51         # 눈 깜빡임 감지를 위한 임계값
MOVING_AVERAGE_WINDOW = 3      # 움직임 평균을 계산하기 위한 윈도우 크기

# 머리 숙임 감지 관련 상수
HEAD_DOWN_ANGLE_THRESHOLD = 15  # 머리가 숙여졌다고 판단하는 각도 임계값 (도 단위)
HEAD_DOWN_WARNING_TIME = 15     # '주의' 상태로 판단하는 머리 숙임 지속 시간 (초 단위)
HEAD_DOWN_DROWSY_TIME = 60      # '졸음' 상태로 판단하는 머리 숙임 지속 시간 (초 단위)

# Face Mesh 랜드마크 인덱스 (MediaPipe Face Mesh의 468개 랜드마크 중 주요 포인트)
NOSE_TIP_INDEX = 1              # 코끝 위치
CHIN_INDEX = 152                # 턱 위치
LEFT_EYE_LEFT_CORNER_INDEX = 33    # 왼쪽 눈의 왼쪽 끝점
RIGHT_EYE_RIGHT_CORNER_INDEX = 133  # 오른쪽 눈의 오른쪽 끝점
LEFT_MOUTH_CORNER_INDEX = 61        # 입의 왼쪽 끝점
RIGHT_MOUTH_CORNER_INDEX = 291      # 입의 오른쪽 끝점

# 눈 감지를 위한 랜드마크 인덱스 그룹
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]   # 왼쪽 눈 윤곽을 구성하는 점들
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]   # 오른쪽 눈 윤곽을 구성하는 점들

# 비디오 처리 관련 변수
frame_count = 0     # 처리된 프레임 수를 추적
quad_data = {}      # 각 ROI(관심 영역)의 상태 데이터를 저장하는 딕셔너리
roi_selector = None # ROI 선택 도구 인스턴스

#-------------------------------------------
# 유틸리티 함수
#-------------------------------------------

def create_quad_data(num_rois):
    """ROI 개수에 따라 상태 추적 데이터 초기화"""
    quad_data = {}
    for i in range(num_rois):
        quad_data[i] = {
            'head_down_start': None,  # 머리 숙임 시작 시간
            'head_down_duration': 0,   # 머리 숙임 지속 시간
        }
    return quad_data

def calculate_head_angle(landmarks):
    """머리 숙임 각도 계산
    
    Args:
        landmarks: MediaPipe Pose 랜드마크 데이터
        
    Returns:
        float: 머리 숙임 각도 (0~90도)
    """
    try:
        # 필요한 랜드마크 추출
        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EAR]

        # 각 랜드마크의 가시성(visibility) 확인
        visibility_threshold = 0.3
        landmarks_visible = (
            nose.visibility > visibility_threshold and
            left_shoulder.visibility > visibility_threshold and
            right_shoulder.visibility > visibility_threshold and
            left_ear.visibility > visibility_threshold and
            right_ear.visibility > visibility_threshold
        )

        if not landmarks_visible:
            # 랜마크가 잘 이지 않으면 엎드린 것으로 간주
            return 90  # 최대 각도 반환

        # 어깨 중심점의 y좌표
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        ear_y = (left_ear.y + right_ear.y) / 2

        # 머리 숙임 정도 계산
        # 코와 어깨, 귀와 어깨의 y좌표 차이 모 려
        nose_shoulder_diff = nose.y - shoulder_y
        ear_shoulder_diff = ear_y - shoulder_y
        
        #  큰 차이값 용
        current_diff = max(nose_shoulder_diff, ear_shoulder_diff)
        
        # 정상 자일 때의 기준값
        reference_diff = 0.2

        # 머 숙임 각도 계산
        head_angle = (current_diff + reference_diff) * 100

        # 디버깅용 출력
        print(f"Nose Y: {nose.y:.3f}, Shoulder Y: {shoulder_y:.3f}, Ear Y: {ear_y:.3f}")
        print(f"Visibility - Nose: {nose.visibility:.2f}, Shoulders: {(left_shoulder.visibility + right_shoulder.visibility)/2:.2f}")
        print(f"Head angle: {head_angle:.1f}")

        return max(0, min(90, head_angle))  # 0~90도 범위로 제한

    except Exception as e:
        print(f"Error calculating head angle: {e}")
        return 0

#-------------------------------------------
# UI 컴포넌트 클래스
#-------------------------------------------

class StartButton(QPushButton):
    """시작 버튼 클래스"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setText('시작')
        self.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 15px 32px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 10px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        self.setFixedSize(200, 50)
        self.setCursor(Qt.PointingHandCursor)  # 마우스 오버 시 커서 변경

class ROISelector:
    """ROI 선택 및 관리"""
    def __init__(self, info_window):
        self.rois = []
        self.drawing = False
        self.current_roi = None
        self.current_name = None
        self.info_window = info_window
        self.start_button = StartButton()
        self.start_button.clicked.connect(self.on_start_clicked)
        self.start_button.hide()
        self.is_ready = False
        # 버튼을 윈도우에 추가
        self.start_button.setParent(None)  # 부모 위젯 제거
        self.start_button.show()  # 버튼 표시

    def start_roi(self, x, y):
        self.drawing = True
        self.current_roi = (x, y, x, y)
        self.current_name = f"ROI_{len(self.rois) + 1}"

    def update_roi(self, x, y):
        if self.drawing:
            x1, y1, _, _ = self.current_roi
            self.current_roi = (x1, y1, x, y)

    def on_start_clicked(self):
        self.is_ready = True
        self.start_button.hide()
        self.start_button.setParent(None)  # 부모 위젯에서 제거

    def finish_roi(self):
        if self.drawing:
            self.rois.append((self.current_roi, self.current_name))
            self.drawing = False
            self.current_roi = None
            self.current_name = None
            
            # ROI 개수에 맞게 quad_data 재생성
            global quad_data
            quad_data = create_quad_data(len(self.rois))
            
            self.info_window.update_roi_count(len(self.rois))
            if len(self.rois) > 0:
                self.start_button.move(50, 50)
                self.start_button.raise_()
                self.start_button.show()

    def draw_rois(self, frame):
        # 저장된 ROI 그리기
        for roi, name in self.rois:
            x1, y1, x2, y2 = roi
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, name, (x1+10, y1+20), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 현재 그리고 는 ROI
        if self.drawing and self.current_roi:
            x1, y1, x2, y2 = self.current_roi
            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def mouse_callback(event, x, y, flags, param):
    roi_selector = param
    if event == cv.EVENT_LBUTTONDOWN:
        roi_selector.start_roi(x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        roi_selector.update_roi(x, y)
    elif event == cv.EVENT_LBUTTONUP:
        roi_selector.finish_roi()

#-------------------------------------------
# UI 클래스
#-------------------------------------------

class StatusUI(QMainWindow):
    """메인 모니터링 UI"""
    def __init__(self, roi_selector):
        super().__init__()
        self.roi_selector = roi_selector
        self.initUI()
        self.status = "수업 중"
        self.current_frame = None

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()

        # 다크 테마 스타 개선
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Arial', sans-serif;
            }
            QFrame {
                background-color: #2d2d2d;
                border-radius: 15px;
                padding: 10px;
            }
        """)

        # 왼쪽 영역: 동영상 표시
        video_frame = QFrame()
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            border: 2px solid #444;
            background-color: #1e1e1e;
            border-radius: 10px;
        """)
        video_layout.addWidget(self.video_label)
        layout.addWidget(video_frame, 2)

        # 오른쪽 레이아웃
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)

        # 상태 현황 섹션
        status_section = QFrame()
        status_layout = QVBoxLayout(status_section)
        
        # 제목
        self.status_title = QLabel("실시간 모니터링 현황")
        self.status_title.setAlignment(Qt.AlignCenter)
        self.status_title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            padding: 15px;
            background-color: #2d2d2d;
            border-radius: 10px;
            margin-bottom: 15px;
        """)
        status_layout.addWidget(self.status_title)

        # 상태 정보
        self.status_message = QLabel()
        self.status_message.setAlignment(Qt.AlignCenter)
        self.status_message.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 20px;
            background-color: #2d2d2d;
            border-radius: 10px;
            margin: 5px;
        """)
        status_layout.addWidget(self.status_message)
        right_layout.addWidget(status_section)

        # 위치 현황 섹션
        location_section = QFrame()
        location_layout = QVBoxLayout(location_section)
        
        self.state_title = QLabel("위치별 상태")
        self.state_title.setAlignment(Qt.AlignCenter)
        self.state_title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            padding: 15px;
            background-color: #2d2d2d;
            border-radius: 10px;
            margin-bottom: 15px;
        """)
        location_layout.addWidget(self.state_title)

        # ROI 표시 영역
        self.roi_label = QLabel()
        self.roi_label.setFixedSize(300, 250)
        self.roi_label.setStyleSheet("""
            background-color: #2d2d2d;
            border: 2px solid #444;
            border-radius: 15px;
            padding: 10px;
        """)
        location_layout.addWidget(self.roi_label)
        right_layout.addWidget(location_section)

        layout.addWidget(right_frame, 1)
        main_widget.setLayout(layout)
        
        self.setWindowTitle('학습 환경 모니터링 시스템')
        self.setGeometry(100, 100, 1400, 800)
        self.show()

    def update_frame(self, frame):
        self.current_frame = frame
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # QLabel 크기 가져오기
        label_width = self.video_label.width()
        label_height = self.video_label.height()

        # 동영상 크기 비율에 맞차서 비디오를 중앙에 발침
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            label_width, label_height, Qt.KeepAspectRatio
        )

        # 비디오 프레임 앙 발침
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

    def update_roi_status(self, person_present, drowsy_status):
        # ROI 수에 맞게 그리드 계산
        num_rois = len(self.roi_selector.rois)
        if num_rois == 0:
            return
        
        # ROI 표시를 위한 그리드 계산
        cols = min(3, num_rois)  # 최대 3열
        rows = (num_rois + cols - 1) // cols  # 행 수 계산
        
        # 각 ROI 칸의 크기 계산
        cell_width = 300 // cols
        cell_height = 250 // rows
        
        # ROI 상태 표시 영역 생성
        roi_pixmap = QPixmap(300, 250)
        roi_pixmap.fill(QColor("#2d2d2d"))
        painter = QPainter(roi_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 지정된 ROI 개수만큼만 처리
        for i in range(num_rois):
            # 주의 상태 확인
            warning_state = False
            if person_present[i] and quad_data[i]['head_down_duration'] >= HEAD_DOWN_WARNING_TIME:
                warning_state = True

            if drowsy_status[i]:
                color = QColor("#ff4444")  # 빨간색 (졸음)
                status = "졸음"
            elif warning_state:
                color = QColor("#ffff44")  # 노란색 (주의)
                status = "주의"
            elif person_present[i]:
                color = QColor("#28a745")  # 초록색 (정상)
                status = "정상"
            else:
                color = QColor("#666666")  # 회색 (부재)
                status = ""

            # ROI 위치 계산
            row = i // cols
            col = i % cols
            x = col * cell_width
            y = row * cell_height
            
            # 그라데이션 효과 추가
            gradient = QLinearGradient(x, y, x + cell_width, y + cell_height)
            gradient.setColorAt(0, color.lighter(120))
            gradient.setColorAt(1, color)
            
            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(Qt.black, 2))
            
            # 둥근 사각형으로 그리기 (여백 추가)
            margin = 2
            painter.drawRoundedRect(
                x + margin, 
                y + margin, 
                cell_width - 2*margin, 
                cell_height - 2*margin, 
                10, 10
            )

            # 텍스트 추가
            painter.setPen(Qt.black if warning_state else Qt.white)  # 주의 상태일 때는 검은색 텍스트
            font = QFont("Arial", 8 if num_rois > 6 else 10, QFont.Bold)
            painter.setFont(font)
            text_rect = QRectF(x, y + cell_height/3, cell_width, cell_height/2)
            painter.drawText(text_rect, Qt.AlignCenter, f"ROI {i+1}\n{status}")

        painter.end()
        self.roi_label.setPixmap(roi_pixmap)

        # 전체 상태 정보 업데이트
        total_people = sum(person_present[:num_rois])
        drowsy_people = sum(drowsy_status[:num_rois])
        warning_people = sum(1 for i in range(num_rois) if person_present[i] and 
                            HEAD_DOWN_WARNING_TIME <= quad_data[i]['head_down_duration'] < HEAD_DOWN_DROWSY_TIME)
        normal_people = total_people - drowsy_people - warning_people

        status_text = (
            f"전체 인원: {total_people}명\n"
            f"{'─' * 20}\n"
            f"정상: {normal_people}명\n"
            f"주의: {warning_people}명\n"
            f"졸음: {drowsy_people}명"
        )

        # 상태에 따른 스타일 설정
        if drowsy_people > 0:
            status_style = """
                font-size: 22px;
                font-weight: bold;
                color: #ffffff;
                background-color: #ff4444;
                border-radius: 10px;
                padding: 20px;
                margin: 5px;
            """
        elif warning_people > 0:
            status_style = """
                font-size: 22px;
                font-weight: bold;
                color: #000000;
                background-color: #ffff44;
                border-radius: 10px;
                padding: 20px;
                margin: 5px;
            """
        else:
            status_style = """
                font-size: 22px;
                font-weight: bold;
                color: #ffffff;
                background-color: #44aa44;
                border-radius: 10px;
                padding: 20px;
                margin: 5px;
            """

        self.status_message.setStyleSheet(status_style)
        self.status_message.setText(status_text)

    def closeEvent(self, event):
        QApplication.quit()
        sys.exit(0)

class InfoWindow(QMainWindow):
    """상세 정보 표시 UI"""
    def __init__(self):
        super().__init__()
        self.initUI()

    def update_roi_count(self, count):
        """ROI 개수에 따라 레이블 동적 생성"""
        # 기존 레이블 제거
        for label in self.info_labels:
            label.setParent(None)
        self.info_labels.clear()

        # 새 레이블 생성
        for i in range(count):
            label = QLabel(f"학생 {i+1}이 자리에 없습니다.")
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setStyleSheet("""
                font-size: 20px;
                font-weight: bold;
                color: #888888;
                background-color: #2d2d2d;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            """)
            self.scroll_layout.addWidget(label)
            self.info_labels.append(label)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout()

        # 스타일 개선
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
                font-family: 'Arial', sans-serif;
                padding: 10px;
                border-radius: 10px;
                margin: 5px;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
                border-radius: 10px;
            }
            QWidget {
                background-color: #1e1e1e;
            }
        """)

        # 제목 레이블 추가
        title_label = QLabel("실시간 학생 상태 모니터링")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #ffffff;
            padding: 15px;
            background-color: #2d2d2d;
            border-radius: 10px;
            margin-bottom: 15px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label)

        # 스크롤 영역 추가
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setSpacing(10)  # 간격 추가
        
        # 보 레이블 리스트 초기화
        self.info_labels = []
        
        scroll_content.setLayout(self.scroll_layout)
        scroll.setWidget(scroll_content)
        self.layout.addWidget(scroll)

        main_widget.setLayout(self.layout)
        self.setWindowTitle('학생별 상태 모니터링')
        self.setGeometry(1550, 100, 600, 800)
        self.show()

    def update_info(self, head_angle, head_direction, region):
        # 상태 판단
        status = "정상"
        details = []
        
        # 머리 숙임 상태 확인
        if head_angle > HEAD_DOWN_ANGLE_THRESHOLD:
            if quad_data[region]['head_down_duration'] >= HEAD_DOWN_DROWSY_TIME:
                status = "졸음 감지"
                details.append(f"머리 숙임 ({int(quad_data[region]['head_down_duration'])}초)")
            elif quad_data[region]['head_down_duration'] >= HEAD_DOWN_WARNING_TIME:
                status = "주의"
                details.append(f"머리 숙임 ({int(quad_data[region]['head_down_duration'])}초)")
        
        # 상태 정보 구성
        info_text = (
            f"▶ 학생 {region + 1} 상태: {status}\n"
            f"{'─' * 40}\n"
            f"머리 각도 정보:\n"
            f"• Pitch: {int(head_angle)}° {'(머리 숙임)' if head_angle > HEAD_DOWN_ANGLE_THRESHOLD else ''}\n"
            f"{'─' * 40}\n"
            f"현재 자세: {head_direction}\n"
            f"특이사항: {', '.join(details) if details else '없음'}\n"
            f"지속시간: {quad_data[region]['head_down_duration']:.1f}초"
        )
        
        # 상태별 스타일 설정
        if status == "졸음 감지":
            style = """
                font-size: 20px;
                font-weight: bold;
                color: #ffffff;
                background-color: #ff4444;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            """
        elif status == "주의":
            style = """
                font-size: 20px;
                font-weight: bold;
                color: #000000;
                background-color: #ffff44;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            """
        else:
            style = """
                font-size: 20px;
                font-weight: bold;
                color: #ffffff;
                background-color: #44aa44;
                border-radius: 10px;
                padding: 15px;
                margin: 5px;
            """
        
        # 스타일과 텍스트 적용
        self.info_labels[region].setStyleSheet(style)
        self.info_labels[region].setText(info_text)

    def reset_info(self, region):
        self.info_labels[region].setText(
            f"▶ 학생 {region + 1}\n"
            f"{'─' * 40}\n"
            f"상태: 자리 비움\n"
            f"{'─' * 40}\n"
            f"특이사항: 없음"
        )
        self.info_labels[region].setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #888888;
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 15px;
            margin: 5px;
        """)

    def closeEvent(self, event):
        QApplication.quit()
        sys.exit(0)

#-------------------------------------------
# 처리 클래스
#-------------------------------------------

class QuadrantProcessor:
    """ROI별 MediaPipe 처리기"""
    def __init__(self, quadrant_id):
        self.quadrant_id = quadrant_id
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            static_image_mode=False
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.head_down_duration = 0

#-------------------------------------------
# 메인 함수
#-------------------------------------------

def detect_person_pose():
    """메인 처리 함수"""
    global roi_selector
    
    # MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    info_window = InfoWindow()
    
    # ROI 선택기 초기화
    roi_selector = ROISelector(info_window)
    
    # StatusUI 초기화 - 여기로 이동
    ui = StatusUI(roi_selector)
    
    cv.namedWindow("Pose Estimation")
    cv.setMouseCallback("Pose Estimation", mouse_callback, roi_selector)
    
    try:
        # 비디오 캡처 초기화
        video_path = "video3.mp4"
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return

        # 비디오 속성 설정
        original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS))
        frame_delay = int(1000/fps)

        # 목표 크기 설정
        target_height = 720
        target_width = int(original_width * (target_height / original_height))

        # 전체 화면 크기 설정
        total_width = target_width + 600
        total_height = max(target_height, 600)

        # 비디오 저장 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output/output_video_{timestamp}.avi"
        
        # output 폴더가 없으면 생성
        os.makedirs("output", exist_ok=True)
        
        # VideoWriter 설정 - 여기로 이동
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (total_width, total_height),
            True
        )

        # 첫 프레임 읽기
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            return
        
        # 프레임 크기 조정
        first_frame = cv.resize(first_frame, (target_width, target_height))
        
        # ROI 선택 모드
        print("ROI를 선택하세요. 선택 완료 후 시작 버튼을 누르세요.")
        print("r: ROI 초기화, ESC: 종료")
        selecting_roi = True
        
        while selecting_roi:
            temp_frame = first_frame.copy()
            roi_selector.draw_rois(temp_frame)
            cv.imshow("Pose Estimation", temp_frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('r'):  # ROI 리셋
                roi_selector.rois = []
                roi_selector.start_button.hide()
                info_window.update_roi_count(0)
                print("ROI가 초기화되었습니다.")
            elif key == 27:  # ESC 키로 종료
                return
            
            # 시작 버튼이 클릭되었는지 확인
            if roi_selector.is_ready:
                selecting_roi = False
                quadrant_processors = [QuadrantProcessor(i) for i in range(len(roi_selector.rois))]
                quad_data = create_quad_data(len(roi_selector.rois))
            
            app.processEvents()

        # 메인 처리 루프
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended")
                break

            # 프레임 크기 조정
            frame = cv.resize(frame, (target_width, target_height))
            img_h, img_w = frame.shape[:2]

            # 상태 변수 초기화
            person_present = [False] * len(roi_selector.rois)
            drowsy_status = [False] * len(roi_selector.rois)

            # ROI 그리기
            roi_selector.draw_rois(frame)

            # 각 ROI에 대해 처리
            for roi_idx, (roi, roi_name) in enumerate(roi_selector.rois):
                x1, y1, x2, y2 = roi
                roi_frame = frame[y1:y2, x1:x2]
                
                if roi_frame.size == 0:
                    continue

                # RGB 변환
                rgb_roi = cv.cvtColor(roi_frame, cv.COLOR_BGR2RGB)
                rgb_roi.flags.writeable = False
                
                # 해당 ROI의 프로세서 사용
                processor = quadrant_processors[roi_idx]
                face_results = processor.face_mesh.process(rgb_roi)
                pose_results = processor.pose.process(rgb_roi)
                
                rgb_roi.flags.writeable = True

                # 사람 감지 로직
                person_detected = False

                # Face Mesh 감지 확인
                if face_results and face_results.multi_face_landmarks:
                    person_detected = True
                    face_landmarks = face_results.multi_face_landmarks[0]
                    # Face Mesh 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        roi_frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )

                # Pose 감지 확인
                if pose_results.pose_landmarks:
                    person_detected = True
                    # Pose 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        roi_frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
                    )

                # 사람 감지 상태 업데이트
                person_present[roi_idx] = person_detected

                # 사람이 감지된 경우에만 상태 정보 업데이트
                if person_detected:
                    if pose_results.pose_landmarks:
                        head_angle = calculate_head_angle(pose_results.pose_landmarks.landmark)
                        
                        # 머리 숙임 상태 처리
                        if head_angle > HEAD_DOWN_ANGLE_THRESHOLD:
                            if quad_data[roi_idx]['head_down_start'] is None:
                                quad_data[roi_idx]['head_down_start'] = time.time()
                            quad_data[roi_idx]['head_down_duration'] = time.time() - quad_data[roi_idx]['head_down_start']
                            
                            if quad_data[roi_idx]['head_down_duration'] >= HEAD_DOWN_DROWSY_TIME:
                                drowsy_status[roi_idx] = True
                                head_direction = "졸음 감지"
                            elif quad_data[roi_idx]['head_down_duration'] >= HEAD_DOWN_WARNING_TIME:
                                head_direction = f"주의 ({int(quad_data[roi_idx]['head_down_duration'])}초)"
                            else:
                                head_direction = f"머리 숙임 감지 ({int(quad_data[roi_idx]['head_down_duration'])}초)"
                        else:
                            quad_data[roi_idx]['head_down_start'] = None
                            quad_data[roi_idx]['head_down_duration'] = 0
                            head_direction = "정상"

                        # 정보 창 업데이트
                        info_window.update_info(head_angle, head_direction, roi_idx)
                else:
                    # 사람이 감지되지 않은 경우 정보 초기화
                    info_window.reset_info(roi_idx)
                    quad_data[roi_idx]['head_down_start'] = None
                    quad_data[roi_idx]['head_down_duration'] = 0

            # UI 업데이트
            ui.update_roi_status(person_present, drowsy_status)
            ui.update_frame(frame)

            # 화면 표시
            cv.imshow("Pose Estimation", frame)

            # 키 입력 처리
            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 또는 ESC로 종료
                print("종료 요청됨")
                break
            elif key == ord('r'):  # r키로 ROI 초기화
                roi_selector.rois = []
                roi_selector.start_button.hide()
                roi_selector.is_ready = False
                info_window.update_roi_count(0)
                print("ROI가 초기화되었습니다.")

            # PyQt 이벤트 처리
            app.processEvents()

    except Exception as e:
        print(f"Error in detect_person_pose: {e}")
        raise e
    finally:
        # 리소스 해제
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        if 'quadrant_processors' in locals():
            for processor in quadrant_processors:
                processor.face_mesh.close()
                processor.pose.close()
        cv.destroyAllWindows()
        
        # PyQt 창 닫기
        if 'ui' in locals():
            ui.close()
        if 'info_window' in locals():
            info_window.close()

#-------------------------------------------
# 프로그램 시작점
#-------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 전역 변수 초기화
    roi_selector = None
    quad_data = {}
    frame_count = 0
    
    try:
        detect_person_pose()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        sys.exit(app.exec_())
