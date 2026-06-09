"""
multi_camera_viewer.py - UI ghép lưới + main loop + CLI (entry point).
======================================================================
Hiển thị 6 webcam USB trên lưới 3x2. Xem mô tả kiến trúc đầy đủ trong config.py.

Chạy:
    python multi_camera_viewer.py            # mở viewer
    python multi_camera_viewer.py discover   # liệt kê cam thật + path (lấy instance_id)
    python multi_camera_viewer.py --detect   # dò index DSHOW thô (fallback)

Phím tắt:  q/ESC thoát | f toàn màn hình | s chụp ảnh lưới (.jpg)
"""

import cv2
import sys
import time
import threading
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from config import (CAMERA_INSTANCE_IDS, CAMERA_INDICES, ROTATE_180_CAMS,
                    CELL_WIDTH, CELL_HEIGHT, GRID_COLS, GRID_ROWS,
                    FREEZE_TIMEOUT, WINDOW_NAME,
                    CAPTURE_MODE, CAMERA_PAIRS, PAIR_DWELL)
from camera import (Camera, list_real_cameras, find_index_by_instance,
                    draw_index_badge, detect_cameras)


# ============================================================
# VẼ CHỮ TIẾNG VIỆT (Unicode) BẰNG PILLOW
# ============================================================
_FONT_CACHE = {}


def _get_font(size):
    """Nạp font hỗ trợ tiếng Việt, cache lại theo size."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def put_text_vn_center(img, text, cy, size, color_bgr):
    """Vẽ chữ tiếng Việt canh giữa theo chiều ngang, tại tung độ cy."""
    font = _get_font(size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (img.shape[1] - tw) // 2
    y = cy - th // 2
    draw.text((x, y), text, font=font,            # PIL dùng RGB nên đảo BGR
              fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ============================================================
# TIỆN ÍCH GHÉP LƯỚI
# ============================================================
def make_disconnected_cell(number):
    """Ô báo chưa kết nối (dòng tiếng Việt canh giữa + badge số). PIL render text
    rất chậm (~5-15ms) -> cache MỘT LẦN lúc khởi động, main chỉ paste."""
    cell = np.zeros((CELL_HEIGHT, CELL_WIDTH, 3), dtype=np.uint8)
    msg = f"Kiểm tra lại kết nối camera {number}"
    cell = put_text_vn_center(cell, msg, CELL_HEIGHT // 2, 24, (60, 60, 240))
    draw_index_badge(cell, number)
    return cell


def build_disconnected_cells():
    """Render trước 6 ô 'mất kết nối' (kèm badge) để main chỉ việc paste."""
    return [make_disconnected_cell(i + 1) for i in range(GRID_COLS * GRID_ROWS)]


def allocate_canvas():
    """Cấp phát MỘT LẦN canvas tổng cho lưới. Tránh np.hstack/vstack mỗi frame
    (mỗi lần gọi sẽ malloc + memcpy toàn bộ ảnh -> tốn CPU & gây jitter)."""
    return np.zeros((GRID_ROWS * CELL_HEIGHT,
                     GRID_COLS * CELL_WIDTH, 3), dtype=np.uint8)


def cell_position(i):
    """Vị trí (row, col) của CAM số (i+1). Bố cục theo CỘT, số CHẴN hàng TRÊN,
    số LẺ hàng DƯỚI:
        CAM2  CAM4  CAM6   (row 0 - trên)
        CAM1  CAM3  CAM5   (row 1 - dưới)
    """
    col = i // 2
    row = 0 if (i + 1) % 2 == 0 else 1   # CAM chẵn -> trên; CAM lẻ -> dưới
    return row, col


def build_grid_into(canvas, cameras, disconnected_cells):
    """Paste cell mới nhất của từng camera vào canvas (main chỉ memcpy, rất nhẹ).
    cam.cell = None (chưa cắm / vừa rớt) -> thay bằng ô 'mất kết nối' đã cache."""
    for i in range(GRID_COLS * GRID_ROWS):
        r, c = cell_position(i)
        y1, y2 = r * CELL_HEIGHT, (r + 1) * CELL_HEIGHT
        x1, x2 = c * CELL_WIDTH, (c + 1) * CELL_WIDTH
        cell = cameras[i].cell if i < len(cameras) else None
        if cell is None:
            cell = disconnected_cells[i]        # đã cache sẵn, không render lại
        canvas[y1:y2, x1:x2] = cell
    return canvas


# ============================================================
# KHỞI TẠO CAMERA + CHƯƠNG TRÌNH CHÍNH
# ============================================================
def build_cameras():
    """Tạo danh sách Camera cho ô 1..N (chỉ số list = CAM (i+1)).
    - Ưu tiên CAMERA_INSTANCE_IDS (ổn định theo cổng USB); in bảng đối chiếu.
    - Để trống / thiếu lib -> fallback CAMERA_INDICES (index DSHOW thủ công)."""
    n_cells = GRID_COLS * GRID_ROWS
    cameras = []
    cams = list_real_cameras() if CAMERA_INSTANCE_IDS else None
    if CAMERA_INSTANCE_IDS and cams is None:
        print("[!] Không liệt kê được camera (thiếu lib) -> fallback CAMERA_INDICES.")
    if CAMERA_INSTANCE_IDS and cams is not None:
        targets = (list(CAMERA_INSTANCE_IDS) + [None] * n_cells)[:n_cells]
        print("Ánh xạ CAM -> instance_id -> index:")
        for i, target in enumerate(targets):
            idx = find_index_by_instance(target, cams or [])
            tail = (target or "").split("\\")[-1]
            print(f"  CAM {i + 1}: ...{tail or '(trống)'} -> "
                  f"{'index ' + str(idx) if idx is not None else 'CHƯA THẤY'}")
            cameras.append(Camera(name=f"CAM {i + 1}", number=i + 1,
                                  cell_size=(CELL_WIDTH, CELL_HEIGHT),
                                  target=target,
                                  rotate180=(i + 1) in ROTATE_180_CAMS))
        return cameras
    # Fallback: index thủ công
    srcs = (list(CAMERA_INDICES) + [None] * n_cells)[:n_cells]
    for i, src in enumerate(srcs):
        cameras.append(Camera(name=f"CAM {i + 1}", number=i + 1,
                              cell_size=(CELL_WIDTH, CELL_HEIGHT), src=src,
                              rotate180=(i + 1) in ROTATE_180_CAMS))
    return cameras


def _paired_capture_loop(cameras, stop_event):
    """[paired] Luân phiên TỪNG CẶP (CAMERA_PAIRS): mở cặp -> chụp ảnh tươi vào ô
    -> giữ PAIR_DWELL giây -> ĐÓNG cả cặp -> sang cặp kế. Chỉ <=2 cam mở cùng lúc
    nên nhẹ USB (hết tràn băng thông) + nhẹ CPU. Chạy 1 thread nền; main thread chỉ
    đọc cam.cell để ghép lưới. Ô cặp chưa tới lượt vẫn giữ ảnh chụp gần nhất."""
    n = len(cameras)
    pairs = [[i for i in pair if 0 <= i < n] for pair in CAMERA_PAIRS]
    while not stop_event.is_set():
        for members in pairs:
            if stop_event.is_set():
                break
            cams = [cameras[i] for i in members]
            for cam in cams:                  # mở (nếu cần) + chụp 1 ảnh tươi
                cam.snapshot()
            stop_event.wait(PAIR_DWELL)        # giữ ảnh cặp này một lúc (vẫn thoát nhanh)
            for cam in cams:                   # ĐÓNG để nhường băng thông cho cặp kế
                cam._release_cap()
    for cam in cameras:                        # dọn khi dừng
        cam._release_cap()


def main():
    print("Đang khởi tạo các camera...")
    cameras = build_cameras()
    paired = CAPTURE_MODE == "paired"
    stop_event = None
    capture_thread = None
    if paired:
        # Chế độ ẢNH luân phiên từng cặp: 1 thread nền tự mở/chụp/đóng từng cặp.
        print("Chế độ PAIRED: luân phiên từng cặp (mỗi lúc chỉ 2 cam mở).")
        stop_event = threading.Event()
        capture_thread = threading.Thread(
            target=_paired_capture_loop, args=(cameras, stop_event), daemon=True)
        capture_thread.start()
    else:
        # Chế độ LIVE: mỗi cam 1 thread đọc riêng. Giãn start để USB negotiate ổn định.
        for cam in cameras:
            cam.start()
            time.sleep(0.4)

    # Pre-allocate canvas + cache ô "mất kết nối" (Pillow text rất chậm).
    canvas = allocate_canvas()
    disconnected_cells = build_disconnected_cells()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False
    print("Bắt đầu hiển thị. Phím: q/ESC thoát | f toàn màn hình | s chụp ảnh")

    # Nhịp ghép lưới + hiển thị. Cap ~30fps để cửa sổ mượt + waitKey gọi đều
    # (Windows không báo "Not Responding").
    frame_period = 1.0 / 30.0
    next_tick = time.time()

    try:
        while True:
            # Watchdog "đứng hình" CHỈ cho chế độ LIVE: cam đang MỞ mà không có
            # khung mới quá FREEZE_TIMEOUT (grab() treo do rung USB) -> ép reset.
            # Chế độ paired tự mở/đóng theo cặp nên không cần (và không được) watchdog.
            if not paired:
                now = time.time()
                for cam in cameras:
                    if (cam.cap is not None and not cam._reset_requested
                            and cam.last_frame_time > 0.0
                            and now - cam.last_frame_time > FREEZE_TIMEOUT):
                        print(f"[watchdog] {cam.name} đứng hình "
                              f"{now - cam.last_frame_time:.1f}s -> ép reset.")
                        cam.request_reset()

            build_grid_into(canvas, cameras, disconnected_cells)
            cv2.imshow(WINDOW_NAME, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord("s"):
                fname = datetime.now().strftime("capture_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(fname, canvas)
                print(f"[+] Đã lưu {fname}")

            # Giữ nhịp display: ngủ phần dư trong period.
            next_tick += frame_period
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()   # tụt nhịp -> reset, không tích nợ thời gian
    finally:
        print("Đang đóng camera...")
        if paired:
            if stop_event is not None:
                stop_event.set()
            if capture_thread is not None:
                capture_thread.join(timeout=3.0)
            for cam in cameras:
                cam._release_cap()
        else:
            for cam in cameras:
                cam.stop()
        cv2.destroyAllWindows()
        print("Đã thoát.")


def discover():
    """In bảng index | name | path của camera THẬT (đã lọc cam ảo) để lấy
    instance_id (nằm trong path) điền vào CAMERA_INSTANCE_IDS."""
    cams = list_real_cameras()
    if cams is None:
        return
    if not cams:
        print("Không thấy camera thật nào (đã cắm cam chưa? có thể toàn cam ảo).")
        return
    print("\n========= CAMERA THẬT PHÁT HIỆN =========")
    for k, (index, name, path) in enumerate(cams, 1):
        print(f"[{k}] index={index}  name={name}")
        print(f"     path = {path}\n")
    print("=> Chép phần instance_id trong path (vd '6&17f9c0cf&0&0000') hoặc cả "
          "path vào CAMERA_INSTANCE_IDS theo thứ tự CAM 1..6.")


if __name__ == "__main__":
    if "discover" in sys.argv:
        discover()
    elif "--detect" in sys.argv:
        detect_cameras()
    else:
        main()
