# 졸음 감지 모니터링 시스템

실시간 영상에서 사용자의 머리 각도를 감지하여 졸음 상태를 모니터링하는 시스템입니다.

## 주요 기능
- 실시간 졸음 상태 감지
- 다중 ROI(관심 영역) 모니터링
- 상태별 통계 및 시각화
- 개별 사용자 상태 추적

## 사용 방법
1. 프로그램 실행 시 첫 프레임에서 ROI 영역을 선택합니다.
2. 마우스 드래그로 모니터링할 영역을 지정합니다.
3. 모든 ROI 선택 후 '시작' 버튼을 클릭합니다.
4. 실시간 모니터링이 시작됩니다.

### 키보드 단축키
| 키 | 기능 |
|---|---|
| `r` | ROI 초기화 |
| `q` 또는 `ESC` | 프로그램 종료 |

## 상태 판단 기준
| 상태 | 조건 |
|---|---|
| 정상 | 머리 숙임 각도 15도 미만 |
| 주의 | 머리 숙임 15초 이상 지속 |
| 졸음 | 머리 숙임 60초 이상 지속 |

## UI 구성
### 1. 메인 화면
- 실시간 영상 표시
- ROI 영역 표시
- 상태 표시 (정상/주의/졸음)

### 2. 상태 모니터링 창
- 전체 인원 현황
- 상태별 인원 통계
- ROI별 상세 정보

### 3. 학생별 상태 창
- 개별 학생 상태 정보
- 머리 각도 정보
- 지속 시간 표시

## 기술 스택
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-latest-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-latest-red)
![PyQt5](https://img.shields.io/badge/PyQt5-latest-yellow)
![NumPy](https://img.shields.io/badge/NumPy-latest-blue)

## 시스템 요구사항
- OS: Windows 10/11, macOS, Linux
- CPU: Intel i5 이상 권장
- RAM: 8GB 이상 권장
- 웹캠 또는 비디오 입력 장치

## 주의사항
- 충분한 조명이 필요합니다.
- 얼굴이 명확히 보이는 각도가 필요합니다.
- ROI는 한 명당 하나의 영역만 지정해야 합니다.

## 문제 해결
### 검은 화면이 표시될 경우
- 비디오 파일 경로 확인
- 카메라 연결 상태 확인

### ROI 선택이 안될 경우
- 프로그램 재시작
- 마우스 이벤트 확인

## 라이선스
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 개발자 정보
- 개발: Shin eunsu
- 이메일: ensoo1015@nate.com
- 버전: 1.0.0

</rewritten_file>
