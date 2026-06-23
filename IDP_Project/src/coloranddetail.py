import cv2
import numpy as np
import imutils
import time

# --- CORE SETTINGS ---
CAMERA_INDEX = 0               # 1 for external Logitech webcams, 0 for internal
CAMERA_BACKEND = cv2.CAP_DSHOW  # High-speed native Windows backend
SMOOTHING_ALPHA = 0.25         # Low-pass filter smoothing coefficient
ANGLE_DEADBAND = 1.5           # Suppresses sub-degree motor/pixel vibrations

# --- ORIENTATION ALLOWANCE ---
ORIENTATION_TOLERANCE = 5.0    # If the part is within ±5 degrees of the trained angle, it's "Desired Orientation"

class AIOrientationInspectionBench:
    def __init__(self):
        self.product_db = {}
        self.pixels_per_mm = 1.0  
        self.sm_w = None
        self.sm_h = None
        self.sm_angle = None

    def get_color_name(self, hue, sat, val):
        if val < 50: return "Black"
        if sat < 40 and val > 150: return "White/Gray"
        if 100 <= hue <= 140: return "Blue"
        if 35 <= hue <= 85: return "Green"
        if 15 <= hue <= 34: return "Yellow/Orange"
        if (0 <= hue <= 14) or (165 <= hue <= 180): return "Red"
        return f"Hue_{int(hue)}"

    def run_bench(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Critical: No active video hardware detected.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.1)
        cap.set(cv2.CAP_PROP_FOCUS, 25) 

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        window_name = "AI Orientation & Alignment Bench"
        cv2.namedWindow(window_name)

        print("\n=== AI ORIENTATION & ALIGNMENT BENCH BOOTED ===")
        print("-> Step 1: Place your item in the PERFECT desired orientation and press 't' to train.")
        print("-> Step 2: Rotate the object on the stage to test the alignment feedback system.\n")

        while True:
            ret, frame = cap.read()
            if not ret: continue

            processing_frame = imutils.resize(frame, width=500)
            hsv_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 60, 160)
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            edged = cv2.dilate(edged, None, iterations=1)

            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            best_contour = None
            max_area = 0
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 1000 and area > max_area:
                    max_area = area
                    best_contour = c

            if best_contour is not None:
                rect = cv2.minAreaRect(best_contour)
                (x, y), (w, h), raw_angle = rect
                
                raw_w = max(w, h)
                raw_h = min(w, h)
                raw_angle = raw_angle + 90 if w < h else raw_angle

                # Anti-Jitter Temporal Filter
                if self.sm_w is None:
                    self.sm_w, self.sm_h, self.sm_angle = raw_w, raw_h, raw_angle
                else:
                    self.sm_w = (SMOOTHING_ALPHA * raw_w) + ((1.0 - SMOOTHING_ALPHA) * self.sm_w)
                    self.sm_h = (SMOOTHING_ALPHA * raw_h) + ((1.0 - SMOOTHING_ALPHA) * self.sm_h)
                    
                    angle_delta = raw_angle - self.sm_angle
                    if angle_delta > 180: angle_delta -= 360
                    elif angle_delta < -180: angle_delta += 360
                    if abs(angle_delta) > ANGLE_DEADBAND:
                        self.sm_angle += (SMOOTHING_ALPHA * angle_delta)

                mm_w = round(self.sm_w / self.pixels_per_mm, 2)
                mm_h = round(self.sm_h / self.pixels_per_mm, 2)
                
                box_area = self.sm_w * self.sm_h
                solidity = round(max_area / box_area, 3) if box_area > 0 else 0.0
                raw_perimeter = cv2.arcLength(best_contour, True)
                mm_perimeter = round(raw_perimeter / self.pixels_per_mm, 2)

                # Color Extraction
                x_mid, y_mid = int(x), int(y)
                y_start, y_end = max(0, y_mid - 8), min(processing_frame.shape[0], y_mid + 8)
                x_start, x_end = max(0, x_mid - 8), min(processing_frame.shape[1], x_mid + 8)
                color_roi = hsv_frame[y_start:y_end, x_start:x_end]
                detected_color = self.get_color_name(np.mean(color_roi[:, :, 0]), np.mean(color_roi[:, :, 1]), np.mean(color_roi[:, :, 2])) if color_roi.size > 0 else "Unknown"

                # Texture Density
                mask = np.zeros(edged.shape, dtype=np.uint8)
                draw_rect = ((x, y), (self.sm_h, self.sm_w), self.sm_angle - 90) if w < h else ((x, y), (self.sm_w, self.sm_h), self.sm_angle)
                box = cv2.boxPoints(draw_rect).astype(np.int32)
                cv2.drawContours(mask, [box], 0, 255, -1)
                detail_density = int(np.sum(cv2.bitwise_and(edged, mask) == 255) / 100)

                # --- CLASSIFIER & ORIENTATION CHECK ENGINE ---
                matched_product = "Searching..."
                orientation_status = "N/A"
                spin_needed = 0.0
                best_match_score = float("inf")
                
                # Default status color: RED (Product is completely unknown)
                hud_color = (0, 0, 255) 

                for name, footprint in self.product_db.items():
                    w_diff = abs(mm_w - footprint["w"]) / footprint["w"]
                    h_diff = abs(mm_h - footprint["h"]) / footprint["h"]
                    
                    if w_diff < 0.12 and h_diff < 0.12 and detected_color == footprint["color"]:
                        solidity_diff = abs(solidity - footprint["solidity"])
                        perimeter_diff = abs(mm_perimeter - footprint["perimeter"]) / footprint["perimeter"]
                        detail_diff = abs(detail_density - footprint["detail"]) / max(1, footprint["detail"])

                        total_score = w_diff + h_diff + solidity_diff + perimeter_diff + detail_diff
                        if total_score < best_match_score and solidity_diff < 0.06:
                            best_match_score = total_score
                            matched_product = name
                            
                            # --- CALCULATE THE ORIENTATION DELTA CORRECTION ---
                            # Subtract current angle from trained baseline golden angle
                            rot_delta = self.sm_angle - footprint["baseline_angle"]
                            if rot_delta > 180: rot_delta -= 360
                            elif rot_delta < -180: rot_delta += 360
                            spin_needed = round(rot_delta, 1)

                            # --- TRAFFIC LIGHT COLOR ENGINE ---
                            if abs(spin_needed) <= ORIENTATION_TOLERANCE:
                                orientation_status = "DESIRED ORIENTATION"
                                hud_color = (0, 255, 0)  # BGR FOR GREEN: Ready to proceed
                            else:
                                orientation_status = "MISALIGNED"
                                hud_color = (0, 255, 255) # BGR FOR YELLOW: Correct part, wrong spin

                # Rendering Bounding Contours and Text Overlays
                cv2.drawContours(processing_frame, [box], 0, hud_color, 2)
                cv2.circle(processing_frame, (x_mid, y_mid), 4, hud_color, -1)

                # Live HUD Telemetry
                cv2.putText(processing_frame, f"ID: {matched_product}", (int(box[0][0]), int(box[0][1]) - 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
                cv2.putText(processing_frame, f"ALIGN: {orientation_status}", (int(box[0][0]), int(box[0][1]) - 47),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 2)
                
                if matched_product != "Searching...":
                    cv2.putText(processing_frame, f"Spin Correction Required: {spin_needed} deg", (int(box[0][0]), int(box[0][1]) - 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200) if hud_color == (0,255,0) else (150, 255, 255), 1)
                cv2.putText(processing_frame, f"Live Angle: {round(self.sm_angle, 1)} deg", (int(box[0][0]), int(box[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            else:
                self.sm_w, self.sm_h, self.sm_angle = None, None, None
                cv2.putText(processing_frame, "PLATFORM EMPTY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(window_name, processing_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): 
                break
            elif key == ord('c'): 
                if self.sm_w is not None:
                    size_input = input("\n[CALIBRATION] Enter object size (mm): ")
                    try:
                        self.pixels_per_mm = self.sm_w / float(size_input)
                        print(f"✔ Spatial Calibration Locked: {round(self.pixels_per_mm, 3)} px/mm")
                    except ValueError: pass
            elif key == ord('t'): 
                if self.sm_w is not None:
                    name_input = input("\n[REGISTRATION] Enter Product Label (This saves current angle as GOLDEN): ")
                    if name_input.strip():
                        self.product_db[name_input.strip()] = {
                            "w": mm_w, "h": mm_h, "solidity": solidity, "perimeter": mm_perimeter,
                            "color": detected_color, "detail": detail_density,
                            "baseline_angle": self.sm_angle  # <--- LOCKS GOLDEN ANGLE POSITION
                        }
                        print(f"✔ Golden Master Profile locked for '{name_input}' at {round(self.sm_angle, 1)} degrees.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    bench = AIOrientationInspectionBench()
    bench.run_bench()