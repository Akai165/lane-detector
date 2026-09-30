import os
import cv2
import numpy as np


def pipeline_threshold(
    img, s_thresh=(170, 255), sx_thresh=(20, 100), l_thresh=(200, 255)
):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    max_sobel = np.max(abs_sobelx)

    if max_sobel > 0:
        scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
    else:
        scaled_sobel = np.zeros_like(l_channel)

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 155) & (b_channel <= 200)] = 1

    combined = np.zeros_like(sxbinary)
    combined[
        (sxbinary == 1) | (s_binary == 1) | (l_binary == 1) | (b_binary == 1)
    ] = 1

    return combined

def estimate_vanishing_point(lines, min_slope=0.5, max_slope=2.5):
    if lines is None:
        return None

    left_lines, right_lines = [], []

    # lines.reshape(-1, 4) normalizza lo shape a N righe x 4 colonne: [x1, y1, x2, y2]
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        if x2 == x1:
            continue  # evita divisione per zero
        
        slope = (y2 - y1) / (x2 - x1)

        # Filtra e separa corsia sinistra e destra
        if -max_slope < slope < -min_slope:
            left_lines.append((slope, y1 - slope * x1))
        elif min_slope < slope < max_slope:
            right_lines.append((slope, y1 - slope * x1))

    if not left_lines or not right_lines:
        return None

    m_left, q_left = np.mean(left_lines, axis=0)
    m_right, q_right = np.mean(right_lines, axis=0)

    if abs(m_left - m_right) < 1e-4:
        return None

    vp_x = int((q_right - q_left) / (m_left - m_right))
    vp_y = int(m_left * vp_x + q_left)
    return vp_x, vp_y


def apply_dynamic_roi(binary_img, prev_vp=None, alpha=0.85):
    h, w = binary_img.shape[:2]

    # Converte la maschera 0/1 in uint8 0/255 per HoughLinesP
    search_area = (binary_img * 255).astype(np.uint8)
    search_area[: int(h * 0.45), :] = 0  # ignora la metà superiore

    lines = cv2.HoughLinesP(
        search_area, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=20
    )
    vp = estimate_vanishing_point(lines)

    # Validazione e stabilizzazione temporale
    default_vp = (w // 2, int(h * 0.58))
    if vp is None or not (0 <= vp[0] <= w and int(h * 0.4) <= vp[1] <= int(h * 0.75)):
        current_vp = prev_vp if prev_vp is not None else default_vp
    else:
        current_vp = (
            (int(alpha * prev_vp[0] + (1 - alpha) * vp[0]),
             int(alpha * prev_vp[1] + (1 - alpha) * vp[1]))
            if prev_vp is not None else vp
        )

    vp_x, vp_y = current_vp
    vertices = np.array([[
        (int(w * 0.08), int(h * 0.88)),
        (vp_x - int(w * 0.04), vp_y + 10),
        (vp_x + int(w * 0.04), vp_y + 10),
        (int(w * 0.92), int(h * 0.82)),
    ]], dtype=np.int32)

    mask = np.zeros_like(binary_img, dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 1)
    masked_img = cv2.bitwise_and(binary_img.astype(np.uint8), mask)

    return masked_img, vertices, current_vp


def apply_roi(binary_img):
    h, w = binary_img.shape[:2]
    vertices = np.array(
        [[
            (int(w * 0.08), int(h * 0.88)),
            (int(w * 0.47), int(h * 0.58)),
            (int(w * 0.53), int(h * 0.58)),
            (int(w * 0.92), int(h * 0.82)),
        ]],
        dtype=np.int32,
    )
    mask = np.zeros_like(binary_img, dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 1)
    masked_img = cv2.bitwise_and(binary_img.astype(np.uint8), mask)
    return masked_img, vertices


def get_perspective_transform_matrice(w, h):
    src = np.float32([
        [w * 0.165, h * 0.96],
        [w * 0.455, h * 0.635],
        [w * 0.545, h * 0.635],
        [w * 0.865, h * 0.96],
    ])
    offset = w * 0.25
    dst = np.float32([
        [offset, h],
        [offset, 0],
        [w - offset, 0],
        [w - offset, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src, dst


def bird_eye_view(binary_img, M):
    h, w = binary_img.shape[:2]
    return cv2.warpPerspective(
        binary_img.astype(np.uint8), M, (w, h), flags=cv2.INTER_LINEAR
    )


# --- Loop Video Principale ---
cap = cv2.VideoCapture("./media/project_video.mp4")
if not cap.isOpened():
    print("Errore: impossibile aprire il file video.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Errore: impossibile leggere il primo frame.")
    cap.release()
    exit()

h, w = frame.shape[:2]
M, Minv, src, dst = get_perspective_transform_matrice(w, h)

HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
OUTPUT_PATH = "./media/output.mp4"
out = None

if not HAS_DISPLAY:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Risoluzione coerente con w, h del video
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 25.0, (w, h))
    print(f"Nessun display trovato (WSL/headless): salvo l'output in '{OUTPUT_PATH}'")

MINIMAP_WIDTH = 370
MINIMAP_HEIGHT = 280
y_offset = 20
x_offset = w - MINIMAP_WIDTH - 20

prev_vp = None
frame_count = 0
while True:
    frame_count += 1

    # 1. Pipeline di sogliatura
    binary_mask = pipeline_threshold(frame)

    # 2. ROI e Bird's Eye View su maschera pulita
    # DECOMMENTA QUA PER ROI STATICA
    #roi_mask, vertices = apply_roi(binary_mask)
    roi_mask, vertices, prev_vp = apply_dynamic_roi(binary_mask, prev_vp)
    bev_mask = bird_eye_view(roi_mask, M)
    bev_visual = cv2.cvtColor(bev_mask * 255, cv2.COLOR_GRAY2BGR)

    # 3. Disegno trapezio ROI direttamente sul frame mostrato
    cv2.polylines(frame, vertices, isClosed=True, color=(0, 255, 0), thickness=2)

    # 4. Minimappa BEV con bordo elegante
    bev_resized = cv2.resize(bev_visual, (MINIMAP_WIDTH, MINIMAP_HEIGHT))
    cv2.rectangle(
        frame,
        (x_offset - 2, y_offset - 2),
        (x_offset + MINIMAP_WIDTH + 1, y_offset + MINIMAP_HEIGHT + 1),
        (0, 255, 255),
        2,
    )
    frame[y_offset : y_offset + MINIMAP_HEIGHT, x_offset : x_offset + MINIMAP_WIDTH] = (
        bev_resized
    )

    # 5. Output
    if HAS_DISPLAY:
        cv2.imshow("Lane Detection con BEV minimap", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    else:
        out.write(frame)
        if frame_count % 50 == 0:
            print(f"  Frame elaborati: {frame_count}")

    ret, frame = cap.read()
    if not ret:
        break

cap.release()
if HAS_DISPLAY:
    cv2.destroyAllWindows()
else:
    out.release()
    print(f"Fatto! Video salvato in '{OUTPUT_PATH}'")