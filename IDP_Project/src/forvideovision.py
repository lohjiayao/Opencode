import cv2
import numpy as np
import imutils
import time
import serial
import serial.tools.list_ports

# --- CORE SETTINGS ---
CAMERA_INDEX = 0               
CAMERA_BACKEND = cv2.CAP_DSHOW  
SMOOTHING_ALPHA = 0.08         
ANGLE_DEADBAND = 2.5           

# --- HARDWARE STEPPER CALCULATIONS ---
STEPS_PER_REV = 1600.0         
DEGREES_PER_STEP = 360.0 / STEPS_PER_REV  

# --- SERIAL CONNECTION PREFERENCES ---
BAUD_RATE = 115200             
SERIAL_COOLDOWN_S = 2.0        

# --- VISION REGION & CAMERA PROFILE TUNING ---
ZOOM_VAL = 30
SHARPNESS_VAL = 255
BOX_X, BOX_Y, BOX_SCALE = 329, 296, 178
W_PCT, H_PCT = 0.70, 0.32  
X_PCT, Y_PCT = 0.55, 0.61  

# --- RELAXED SPATIAL TOLERANCES FOR ROTATION ---
DIMENSION_TOLERANCE = 0.20     # Increased from 0.12 to 0.20 to handle rotation pixel distortion
SOLIDITY_TOLERANCE = 0.12      # Increased from 0.06 to 0.12
ORIENTATION_TOLERANCE = 5.0    

class AIOrientationInspectionBench:
    def __init__(self):
        self.product_db = {}
        self.pixels_per_mm = 1.0  
        self.sm_w = None
        self.sm_h = None
        self.sm_angle = None
        self.last_transmission_time = 0.0
        
        # --- AUTO DETECT COMPORT ---
        self.ser = None
        ports = list(serial.tools.list_ports.comports())
        for port in ports:
            if any(k in port.description for k in ["Silicon Labs", "USB-to-UART", "CH340", "Arduino", "ESP32"]):
                try:
                    self.ser = serial.Serial(port.device, BAUD_RATE, timeout=1)
                    print(f"[SERIAL] Connected directly to Hardware on port: {port.device}")
                    break
                except Exception as e:
                    print(f"[SERIAL] Port found ({port.device}) but binding failed: {e}")
        
        if self.ser is None:
            print("[SERIAL WARNING] No active microcontrollers discovered. Running in simulation mode.")

    def get_color_name(self, hue, sat, val):
        if val < 50: return "Black"
        if sat < 40 and val > 150: return "White/Gray"
        if 100 <= hue <= 140: return "Blue"
        if 35 <= hue <= 85: return "Green"
        if 15 <= hue <= 34: return "Yellow/Orange"
        if (0 <= hue <= 14) or (165 <= hue <= 180): return "Red"
        return f"Hue_{int(hue)}"

    def send_step_command(self, angular_error):
        if self.ser is None:
            return
            
        current_time = time.time()
        if (current_time - self.last_transmission_time) < SERIAL_COOLDOWN_S:
            return 
            
        physical_steps_needed = int(round(angular_error / DEGREES_PER_STEP))
        
        if abs(physical_steps_needed) > 0:
            command_string = f"STEPS:{physical_steps_needed}\n"
            print(f"[HARDWARE ACTION] Target drift detected ({angular_error}°). Issuing {physical_steps_needed} steps down the link.")
            try:
                self.ser.write(command_string.encode('utf-8'))
                self.last_transmission_time = current_time
            except Exception as e:
                print(f"[SERIAL ERROR] Failed to send hardware instructions: {e}")

    def run_bench(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Critical Error: Camera hardware offline.")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.1)
        cap.set(cv2.CAP_PROP_FOCUS, 25) 

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        window_name = "AI Orientation & Alignment Bench"
        cv2.namedWindow(window_name)

        print("\n=== AI INSPECTION BENCH ONLINE ===")

        while True:
            ret, frame = cap.read()
            if not ret: continue

            cap.set(cv2.CAP_PROP_ZOOM, float(ZOOM_VAL))
            cap.set(cv2.CAP_PROP_SHARPNESS, float(SHARPNESS_VAL))

            half = BOX_SCALE // 2
            x1 = max(0, min(BOX_X - half, 640 - BOX_SCALE))
            y1 = max(0, min(BOX_Y - half, 480 - BOX_SCALE))
            zoom_crop = frame[y1:y1+BOX_SCALE, x1:x1+BOX_SCALE]

            processing_frame = cv2.resize(zoom_crop, (640, 480))
            cw, ch = 640, 480
            
            rect_w, rect_h = int(cw * W_PCT), int(ch * H_PCT)
            start_x = int(X_PCT * cw) - (rect_w // 2)
            start_y = int(Y_PCT * ch) - (rect_h // 2)
            
            roi_x1, roi_y1 = max(0, start_x), max(0, start_y)
            roi_x2, roi_y2 = min(cw, start_x + rect_w), min(ch, start_y + rect_h)
            
            roi = processing_frame[roi_y1:roi_y2, roi_x1:roi_x2]
            hsv_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2HSV)
            
            if roi.size > 0:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, 90]) 
                black_mask = cv2.inRange(roi_hsv, lower_black, upper_black)
                
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                masked_gray = cv2.bitwise_and(gray, gray, mask=black_mask)
                
                low_blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
                sharpened = cv2.addWeighted(masked_gray, 2.2, low_blur, -1.2, 0)
                
                blurred_final = cv2.GaussianBlur(sharpened, (5, 5), 0)
                edged = cv2.Canny(blurred_final, 50, 150)
                edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
                edged = cv2.dilate(edged, None, iterations=1)

                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
            else:
                cnts = []

            best_contour, max_area = None, 0
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 300 and area > max_area: 
                    max_area = area
                    best_contour = c

            if best_contour is not None:
                shifted_contour = best_contour.copy()
                shifted_contour[:, :, 0] += roi_x1
                shifted_contour[:, :, 1] += roi_y1

                rect = cv2.minAreaRect(shifted_contour)
                (x, y), (w_box, h_box), raw_angle = rect
                
                raw_w, raw_h = max(w_box, h_box), min(w_box, h_box)
                raw_angle = raw_angle + 90 if w_box < h_box else raw_angle

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
                raw_perimeter = cv2.arcLength(shifted_contour, True)
                mm_perimeter = round(raw_perimeter / self.pixels_per_mm, 2)

                x_mid, y_mid = int(x), int(y)
                y_start, y_end = max(0, y_mid - 8), min(processing_frame.shape[0], y_mid + 8)
                x_start, x_end = max(0, x_mid - 8), min(processing_frame.shape[1], x_mid + 8)
                color_roi = hsv_frame[y_start:y_end, x_start:x_end]
                detected_color = self.get_color_name(np.mean(color_roi[:, :, 0]), np.mean(color_roi[:, :, 1]), np.mean(color_roi[:, :, 2])) if color_roi.size > 0 else "Unknown"

                mask = np.zeros((processing_frame.shape[0], processing_frame.shape[1]), dtype=np.uint8)
                draw_rect = ((x, y), (self.sm_h, self.sm_w), self.sm_angle - 90) if w_box < h_box else ((x, y), (self.sm_w, self.sm_h), self.sm_angle)
                box = cv2.boxPoints(draw_rect).astype(np.int32)
                cv2.drawContours(mask, [box], 0, 255, -1)
                
                full_edged = np.zeros_like(mask)
                full_edged[roi_y1:roi_y2, roi_x1:roi_x2] = edged
                detail_density = int(np.sum(cv2.bitwise_and(full_edged, mask) == 255) / 100)

                matched_product = "Searching..."
                orientation_status = "N/A"
                spin_needed = 0.0
                best_match_score = float("inf")
                hud_color = (0, 0, 255) 

                # MATCH LOOP WITH AUTOMATIC SPATIAL TOLERANCE FALLBACKS
                for name, footprint in self.product_db.items():
                    w_diff = abs(mm_w - footprint["w"]) / footprint["w"]
                    h_diff = abs(mm_h - footprint["h"]) / footprint["h"]
                    
                    # Check dimensions with expanded rotation allowance
                    if w_diff < DIMENSION_TOLERANCE and h_diff < DIMENSION_TOLERANCE and detected_color == footprint["color"]:
                        solidity_diff = abs(solidity - footprint["solidity"])
                        perimeter_diff = abs(mm_perimeter - footprint["perimeter"]) / footprint["perimeter"]
                        detail_diff = abs(detail_density - footprint["detail"]) / max(1, footprint["detail"])

                        # Weighted scoring prioritizing structural bounds
                        total_score = (w_diff * 2.0) + (h_diff * 2.0) + solidity_diff + perimeter_diff
                        
                        if total_score < best_match_score and solidity_diff < SOLIDITY_TOLERANCE:
                            best_match_score = total_score
                            matched_product = name
                            
                            rot_delta = self.sm_angle - footprint["baseline_angle"]
                            if rot_delta > 180: rot_delta -= 360
                            elif rot_delta < -180: rot_delta += 360
                            spin_needed = round(rot_delta, 1)

                            if abs(spin_needed) <= ORIENTATION_TOLERANCE:
                                orientation_status = "DESIRED ORIENTATION"
                                hud_color = (0, 255, 0)  
                            else:
                                orientation_status = "MISALIGNED"
                                hud_color = (0, 255, 255) 
                                self.send_step_command(spin_needed)

                cv2.drawContours(processing_frame, [box], 0, hud_color, 2)
                cv2.circle(processing_frame, (x_mid, y_mid), 4, hud_color, -1)
                cv2.rectangle(processing_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 1)

                cv2.putText(processing_frame, f"ID: {matched_product}", (int(box[0][0]), int(box[0][1]) - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud_color, 2)
                cv2.putText(processing_frame, f"ALIGN: {orientation_status}", (int(box[0][0]), int(box[0][1]) - 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 2)
                
                if matched_product != "Searching...":
                    cv2.putText(processing_frame, f"Spin Correction Required: {spin_needed} deg", (int(box[0][0]), int(box[0][1]) - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200) if hud_color == (0,255,0) else (150, 255, 255), 1)
                cv2.putText(processing_frame, f"Live Angle: {round(self.sm_angle, 1)} deg", (int(box[0][0]), int(box[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            else:
                self.sm_w, self.sm_h, self.sm_angle = None, None, None
                cv2.putText(processing_frame, "TARGET OBJECT EMPTY / WAITING FOR BLACK CHIP", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(processing_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 0, 255), 1)

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
                    name_input = input("\n[REGISTRATION] Enter Product Label/Name: ")
                    if name_input.strip():
                        product_name = name_input.strip()
                        self.product_db[product_name] = {
                            "w": mm_w, 
                            "h": mm_h, 
                            "solidity": solidity, 
                            "perimeter": mm_perimeter,
                            "color": detected_color, 
                            "detail": detail_density,
                            "baseline_angle": self.sm_angle 
                        }
                        print(f"✔ Golden Master Profile locked for '{product_name}' at {round(self.sm_angle, 1)} degrees baseline.")
                else:
                    print("\n[TRAINING FAILED] Cannot register profile. Ensure a valid black chip is visible inside the box boundary.")

        if self.ser and self.ser.is_opened:
            self.ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    bench = AIOrientationInspectionBench()
    bench.run_bench()