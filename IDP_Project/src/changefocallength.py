import cv2
import numpy as np
import imutils
import time

# --- CORE SETTINGS ---
CAMERA_INDEX = 1               # 1 for external Logitech webcams, 0 for internal
CAMERA_BACKEND = cv2.CAP_DSHOW  # High-speed native Windows backend
SMOOTHING_ALPHA = 0.25         # Low-pass filter smoothing coefficient
ANGLE_DEADBAND = 1.5           # Suppresses sub-degree motor/pixel vibrations

class VisionTestingBench:
    def __init__(self):
        # Component Fingerprint Database
        self.product_db = {}
        
        # Pixels-to-Millimeters Calibration (Press 'c' to set)
        self.pixels_per_mm = 1.0  
        
        # Low-pass filter memory
        self.sm_w = None
        self.sm_h = None
        self.sm_angle = None

    def run_bench(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
        if not cap.isOpened():
            print(f"Failed index {CAMERA_INDEX}, falling back to index 0...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("Critical: No active video hardware detected.")
                return

        # Lock high-quality macro focus for Logitech hardware
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        time.sleep(0.1)
        cap.set(cv2.CAP_PROP_FOCUS, 25) # Change this number if your blue path goes blurry

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        window_name = "AI Vision Test Bench (No Dashboard)"
        cv2.namedWindow(window_name)

        print("\n=== AI VISION TEST BENCH BOOTED ===")
        print("-> Press 'c' to Calibrate scale using an item of known size.")
        print("-> Press 't' to Train/Register a new Samtec product footprint.")
        print("-> Press 'q' to safely terminate the video link.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Hardware drop frame encountered.")
                continue

            # --- COMPUTER VISION INITIAL PIPELINE ---
            processing_frame = imutils.resize(frame, width=500)
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

            # --- GEOMETRIC CORRECTION & INTERPOLATION ---
            if best_contour is not None:
                rect = cv2.minAreaRect(best_contour)
                (x, y), (w, h), raw_angle = rect
                
                raw_w = max(w, h)
                raw_h = min(w, h)
                raw_angle = raw_angle + 90 if w < h else raw_angle

                # Anti-Jitter Moving Average Filter
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

                # Real-world metric parsing
                mm_w = round(self.sm_w / self.pixels_per_mm, 2)
                mm_h = round(self.sm_h / self.pixels_per_mm, 2)
                
                # --- NEW SHAPE ANALYSIS TIERS ---
                box_area = self.sm_w * self.sm_h
                solidity = round(max_area / box_area, 3) if box_area > 0 else 0.0
                raw_perimeter = cv2.arcLength(best_contour, True)
                mm_perimeter = round(raw_perimeter / self.pixels_per_mm, 2)

                # --- TARGET RECOGNITION CLASSIFIER ---
                matched_product = "Searching..."
                best_match_score = float("inf")
                match_color = (0, 0, 255) # Red for unknown target profiles

                for name, footprint in self.product_db.items():
                    # Check box lengths and widths (10% standard tolerance threshold)
                    w_diff = abs(mm_w - footprint["w"]) / footprint["w"]
                    h_diff = abs(mm_h - footprint["h"]) / footprint["h"]
                    
                    # Check interior solidity and perimeter outline complexity
                    solidity_diff = abs(solidity - footprint["solidity"])
                    perimeter_diff = abs(mm_perimeter - footprint["perimeter"]) / footprint["perimeter"]

                    if w_diff < 0.10 and h_diff < 0.10:
                        # Tie-breaker computation score combines dimensions + topology
                        total_score = w_diff + h_diff + solidity_diff + perimeter_diff
                        if total_score < best_match_score and solidity_diff < 0.05:
                            best_match_score = total_score
                            matched_product = name
                            match_color = (0, 255, 0) # Green for verified database hit

                # Rendering Bounding Overlays
                if w < h:
                    draw_rect = ((x, y), (self.sm_h, self.sm_w), self.sm_angle - 90)
                else:
                    draw_rect = ((x, y), (self.sm_w, self.sm_h), self.sm_angle)
                
                box = cv2.boxPoints(draw_rect).astype(np.int32)
                cv2.drawContours(processing_frame, [box], 0, match_color, 2)

                # HUD Display Output
                cv2.putText(processing_frame, f"ID: {matched_product}", (int(box[0][0]), int(box[0][1]) - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, match_color, 2)
                cv2.putText(processing_frame, f"Dim: {mm_w}x{mm_h} mm", (int(box[0][0]), int(box[0][1]) - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
                cv2.putText(processing_frame, f"Solidity: {solidity} | Perim: {mm_perimeter}mm", (int(box[0][0]), int(box[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            else:
                self.sm_w, self.sm_h, self.sm_angle = None, None, None
                cv2.putText(processing_frame, "STAGE EMPTY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Control Status Overlays
            cv2.putText(processing_frame, f"Scale: {round(self.pixels_per_mm, 2)} px/mm", (20, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(processing_frame, f"Profiles Registered: {len(self.product_db)}", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(window_name, processing_frame)
            key = cv2.waitKey(1) & 0xFF

            # --- LIVE KEYBOARD HANDLERS ---
            if key == ord('q'): # Quit
                break
                
            elif key == ord('c'): # Calibrate 
                if self.sm_w is not None:
                    size_input = input("\n[CALIBRATION] Enter real millimeter length of current target: ")
                    try:
                        real_mm = float(size_input)
                        self.pixels_per_mm = self.sm_w / real_mm
                        print(f"✔ Calibration Locked: {round(self.pixels_per_mm, 3)} pixels/mm")
                    except ValueError:
                        print("✖ Invalid numeric input.")
                else:
                    print("✖ Calibration failed: No product detected on stage.")

            elif key == ord('t'): # Train Shape Profile
                if self.sm_w is not None:
                    name_input = input("\n[REGISTRATION] Enter unique label for this Samtec product profile: ")
                    if name_input.strip():
                        # Fingerprint incorporates shape complexity to isolate items with matching bounding boxes
                        self.product_db[name_input.strip()] = {
                            "w": mm_w,
                            "h": mm_h,
                            "solidity": solidity,
                            "perimeter": mm_perimeter
                        }
                        print(f"✔ Profile '{name_input}' locked with complex shape tracking dimensions.")
                else:
                    print("✖ Training failed: Stage empty.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    bench = VisionTestingBench()
    bench.run_bench()
