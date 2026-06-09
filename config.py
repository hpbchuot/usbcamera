"""
config.py - Cấu hình + mô tả kiến trúc Multi-Camera Viewer (6 webcam USB 720p)
==============================================================================
Module hiển thị 6 webcam USB trên lưới 3x2, MỖI CAM 1 THREAD RIÊNG:
- Mỗi cam có 1 thread daemon liên tục grab() để XẢ buffer DirectShow nên khung
  luôn TƯƠI (trễ ~1-2 frame). Chỉ retrieve()+render theo nhịp DISPLAY_FPS để nhẹ
  CPU (grab() rẻ vì không decode).
- Mỗi cam ghi vào Ô CỦA RIÊNG NÓ -> 1 cam mất/treo KHÔNG ảnh hưởng ô cam khác;
  tự mở lại trong chính thread của nó nên không kẹt cam khác.
- Ép codec MJPG để tiết kiệm băng thông USB; in codec/độ phân giải THỰC TẾ lúc mở.

ĐỊNH DANH CAMERA THEO INSTANCE_ID (ổn định theo cổng USB vật lý):
- Mỗi CAM khớp theo instance_id (CAMERA_INSTANCE_IDS) bằng cv2_enumerate_cameras
  -> miễn nhiễm với việc index DSHOW bị đảo do có camera ảo (ManyCam) / cắm lại.
- Lấy instance_id:  python multi_camera_viewer.py discover

BỐ CỤC LƯỚI 3x2 (số CHẴN hàng TRÊN, số LẺ hàng DƯỚI, theo cột):
        CAM2   CAM4   CAM6
        CAM1   CAM3   CAM5

Phím tắt:  q/ESC thoát | f toàn màn hình | s chụp ảnh lưới (.jpg)
Cài đặt:   pip install opencv-python numpy pillow cv2-enumerate-cameras

Tách 3 file:
- config.py              : cấu hình (file này)
- camera.py              : engine - định danh + lớp Camera đa luồng + vẽ cell
- multi_camera_viewer.py : UI ghép lưới + main loop + CLI (entry point)
"""

# ============================================================
# ĐỊNH DANH CAMERA THEO INSTANCE_ID (ổn định theo cổng USB vật lý)
# ============================================================
# Mỗi CAM khớp theo device instance_id (Device Manager -> Details -> "Device
# instance path", hoặc chạy: python multi_camera_viewer.py discover). Khớp bằng
# so-substring đã chuẩn hóa với 'path' của cv2_enumerate_cameras nên miễn nhiễm
# khác biệt dấu \ vs # và hoa/thường. Index DSHOW có thể đảo khi có camera ảo /
# cắm lại, nhưng instance_id cố định theo cổng -> luôn đúng ô.
#
# Thứ tự CAM 1..6 ánh xạ ra lưới (số CHẴN hàng TRÊN, số LẺ hàng DƯỚI):
#       CAM2   CAM4   CAM6   (hàng trên)
#       CAM1   CAM3   CAM5   (hàng dưới)
# Dùng raw-string (r"...") vì instance_id có dấu \.
CAMERA_INSTANCE_IDS = [
    r"USB\VID_4C4A&PID_4A55&MI_00\6&17f9c0cf&0&0000",  # CAM 1 -> dưới-trái
    r"USB\VID_4C4A&PID_4A55&MI_00\6&540102a&0&0000",   # CAM 2 -> trên-trái
    r"USB\VID_4C4A&PID_4A55&MI_00\6&2e21298c&0&0000",  # CAM 3 -> dưới-giữa
    r"USB\VID_4C4A&PID_4A55&MI_00\6&1b6778e7&0&0000",  # CAM 4 -> trên-giữa
    r"USB\VID_4C4A&PID_4A55&MI_00\7&1543a2f&0&0000",   # CAM 5 -> dưới-phải
    r"USB\VID_4C4A&PID_4A55&MI_00\7&2a355391&0&0000",   # CAM 6 -> trên-phải
]

# Từ khóa tên camera ẢO cần loại khi liệt kê (so khớp không phân biệt hoa/thường).
VIRTUAL_KEYWORDS = ["manycam", "obs", "virtual", "xsplit",
                    "snap camera", "droidcam", "splitcam", "e2esoft", "iriun"]

# Fallback khi CAMERA_INSTANCE_IDS để trống ([]) hoặc thiếu thư viện enumerate:
# dùng index DSHOW thủ công cho từng ô 1..6 (dò bằng --detect).
CAMERA_INDICES = [0, 1, 2, 3, 4, 5]

# Các CAM (theo SỐ thứ tự 1..6) cần xoay khung hình 180° (cam lắp ngược).
ROTATE_180_CAMS = [1, 3, 5]

# ============================================================
# ĐỘ PHÂN GIẢI / LƯỚI / NHỊP
# ============================================================
# Độ phân giải + codec bắt từ camera (nguồn).
# 640x480 + YUY2 (raw): YUY2 KHÔNG nén -> KHÔNG tốn CPU giải mã (bỏ được "sàn
# decode" của MJPEG) => CPU nhẹ hơn NHIỀU. Đổi lại NẶNG băng thông USB
# (~147 Mbps/cam @640x480@30). 4 cam qua hub ≈ 590 Mbps -> CẦN hub USB 3.0
# (USB 2.0 ~480 Mbps sẽ nghẽn: tụt FPS/rớt khung). Đổi CAPTURE_FOURCC="MJPG"
# (nén, nhẹ USB nhưng nặng CPU decode) nếu hub không gánh nổi raw.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30
CAPTURE_FOURCC = "YUY2"

# Kích thước MỖI Ô hiển thị (16:9). 640x360 x lưới 3x2 => cửa sổ 1920x720.
CELL_WIDTH = 640
CELL_HEIGHT = 360

# Lưới 3 cột x 2 hàng = 6 ô.
GRID_COLS = 3
GRID_ROWS = 2

# Nhịp decode + render mỗi cam (grab() vẫn chạy liên tục để xả buffer).
DISPLAY_FPS = 10

# Ghìm nhịp vòng grab (DIỆT BUSY-SPIN). grab() trên driver này trả về tức thì
# (không block) -> vòng while quay hàng triệu lần/giây × 6 thread -> CPU full.
# Ghìm ~ ĐÚNG tốc độ camera sinh khung (30) để vẫn xả buffer kịp (trễ thấp) mà
# không spin. Hạ xuống 20 nếu cần bớt CPU thêm (chấp nhận trễ nhỉnh hơn).
GRAB_FPS = 30

# Số lần grab()/retrieve() lỗi LIÊN TIẾP trước khi coi là mất kết nối. Dung sai
# này tránh rớt oan khi grab() trả False thoáng qua (rung USB) lúc đang pacing.
GRAB_FAIL_LIMIT = 8

# Cooldown thử MỞ LẠI 1 camera đang mất (giây). Mở DSHOW cam đã rớt có thể block
# ~1-3s; giãn ra để thread cam đó không thử mở dồn dập (cam khác không bị ảnh hưởng).
RECONNECT_INTERVAL = 5.0

# Watchdog "đứng hình": cam đang MỞ mà không có khung mới quá ngần này giây (vd USB
# treo do rung khiến grab() block, không trả False) -> ép reset. Đặt > vài chu kỳ
# grab bình thường để không reset oan.
FREEZE_TIMEOUT = 3.0

WINDOW_NAME = "Multi-Camera Viewer (6 CAM)"
