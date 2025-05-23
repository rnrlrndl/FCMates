# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
import time

import cv2
import numpy as np
import torch

# --- 설정값 ---
MODEL_TYPE     = "MiDaS_small"
CALIB_FRAMES   = 60          # 캘리브레이션에 사용할 프레임 수
# 실행 시 사용자 입력으로 기준 거리(미터 단위)를 받습니다.
KNOWN_DISTANCE = float(input("캘리브레이션 기준 거리를 미터 단위로 입력하세요 (예: 0.04): "))
DEPTH_ALPHA    = 0.2         # 깊이 EMA 계수
SPEED_BETA     = 0.3         # 속도 EMA 계수
SPEED_THRESH   = 0.01        # 속도 클리핑 임계값 (m/s)
PATCH_SIZE     = 50          # 중앙 패치 크기 (픽셀)
INPUT_SIZE     = 384         # MiDaS small 기본 입력 크기
# ----------------

def init_depth_model(model_type):
    """MiDaS 모델 로드 및 설정"""
    print(f"[INFO] 모델 로딩: {model_type}")
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=torch.float32).eval()
    return model, device

def preprocess(frame):
    """OpenCV/NumPy로 직접 구현한 MiDaS 전처리 (HWC→CHW)"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32)
    std  = np.array([0.229,0.224,0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2,0,1).unsqueeze(0)

def measure_raw_depth(cap, model, device):
    """캘리브레이션용 raw_depth 평균값 계산"""
    vals = []
    for _ in range(CALIB_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue
        inp = preprocess(frame).to(device)
        with torch.no_grad():
            pred = model(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()
        h, w = pred.shape
        cy, cx = h//2, w//2
        patch = pred[cy-PATCH_SIZE//2:cy+PATCH_SIZE//2,
                     cx-PATCH_SIZE//2:cx+PATCH_SIZE//2]
        vals.append(np.median(patch))
    return float(np.mean(vals))

def main():
    print("[INFO] 프로그램 시작")
    model, device = init_depth_model(MODEL_TYPE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 카메라 열기 실패")
        sys.exit(1)

    # 1) 자동 캘리브레이션
    print(f"[INFO] {CALIB_FRAMES}프레임으로 캘리브레이션 중...")
    raw_ref = measure_raw_depth(cap, model, device)
    DIST_FACTOR = KNOWN_DISTANCE * raw_ref
    print(f"[INFO] raw_ref={raw_ref:.3f}, DIST_FACTOR={DIST_FACTOR:.3f}")

    smoothed_depth  = None
    smoothed_speed  = 0.0
    prev_time       = None
    prev_distance_m = None

    print("[INFO] 'q' 키로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 읽기 실패, 루프 종료")
            break
        now = time.time()

        # 2) 깊이 추정
        inp = preprocess(frame).to(device)
        with torch.no_grad():
            pred = model(inp)
            depth_map = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # 중앙 패치 median → raw_depth
        h, w = depth_map.shape
        cy, cx = h//2, w//2
        patch = depth_map[cy-PATCH_SIZE//2:cy+PATCH_SIZE//2,
                          cx-PATCH_SIZE//2:cx+PATCH_SIZE//2]
        raw_depth = float(np.median(patch))

        # EMA 깊이 스무딩
        if smoothed_depth is None:
            smoothed_depth = raw_depth
        else:
            smoothed_depth = DEPTH_ALPHA * raw_depth + (1 - DEPTH_ALPHA) * smoothed_depth

        # 실제 거리 계산 (반전 방식)
        distance_m = DIST_FACTOR / smoothed_depth

        # 속도 계산 (거리 변화량 기준)
        if prev_time is not None:
            dt = now - prev_time
            speed_m_s = (prev_distance_m - distance_m) / dt
        else:
            speed_m_s = 0.0

        # EMA 속도 스무딩 및 임계값 적용
        smoothed_speed = SPEED_BETA * speed_m_s + (1 - SPEED_BETA) * smoothed_speed
        if abs(smoothed_speed) < SPEED_THRESH:
            smoothed_speed = 0.0

        # 상태 업데이트
        prev_time       = now
        prev_distance_m = distance_m

        # 결과 화면 출력
        cv2.putText(frame, f"Dist: {distance_m:.3f} m", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(frame, f"Speed: {smoothed_speed*3.6:.1f} km/h", (10,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Depth & Speed", frame)
        if cv2.waitKey(1) == ord('q'):
            print("[INFO] 종료키(q) 입력—종료")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
