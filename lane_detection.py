import cv2
import numpy as np
import time

class SteeringFilter:
    def __init__(self, smooth_factor=0.9):
        self.prev_angle = 0
        self.prev_offset = 0
        self.smooth_factor = smooth_factor
    
    def update(self, new_angle, new_offset):
        # Smooth the steering angle and offset using exponential moving average
        smooth_angle = (self.smooth_factor * self.prev_angle + 
                       (1 - self.smooth_factor) * new_angle)
        smooth_offset = (self.smooth_factor * self.prev_offset + 
                       (1 - self.smooth_factor) * new_offset)
        
        self.prev_angle = smooth_angle
        self.prev_offset = smooth_offset
        
        return smooth_angle, smooth_offset

def calculate_lane_center(lines, frame_width, frame_height):
    if lines is None:
        return frame_width // 2
    
    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.1:
            continue
            
        if slope != 0:
            x_bottom = int(x1 + (frame_height - y1) / slope)
            if x_bottom < frame_width // 2:
                left_lines.append(x_bottom)
            else:
                right_lines.append(x_bottom)
    
    if left_lines and right_lines:
        left_x = max(left_lines)
        right_x = min(right_lines)
        center = (left_x + right_x) // 2
    elif left_lines:
        center = max(left_lines) + 200
    elif right_lines:
        center = min(right_lines) - 200
    else:
        center = frame_width // 2
        
    return center

def calculate_steering_angle(frame_center, lane_center, frame_width):
    offset = lane_center - frame_center
    max_offset = frame_width // 6
    max_angle = 30
    
    steering_angle = (offset / max_offset) * max_angle
    steering_angle = max(-max_angle, min(max_angle, steering_angle))
    
    return steering_angle, offset

def process_frame(frame, steering_filter):
    height, width = frame.shape[:2]
    frame_center = width // 2
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, binary = cv2.threshold(blurred, 190, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    # Adjust the ROI to include farther distances
    roi_vertices = np.array([
        [(0, height),
         (0, height * 0.3),  # Increased ROI to start higher
         (width, height * 0.3),
         (width, height)]], dtype=np.int32)
    
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked = cv2.bitwise_and(binary, mask)
    
    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=40,
        maxLineGap=20
    )
    
    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) > 20 and abs(angle) < 160:
                # Use darker green for the overlay
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 100, 0), 30)
    
    lane_center = calculate_lane_center(lines, width, height)
    raw_steering_angle, raw_offset = calculate_steering_angle(frame_center, lane_center, width)
    
    # Apply smoothing filter
    steering_angle, offset = steering_filter.update(raw_steering_angle, raw_offset)
    
    # Visualization
    cv2.line(line_image, (frame_center, height), (frame_center, height - 100), (255, 0, 0), 2)
    cv2.line(line_image, (lane_center, height), (lane_center, height - 100), (0, 100, 0), 2)  # Darker green
    
    offset_color = (0, 255, 0) if abs(offset) < 20 else (0, 165, 255) if abs(offset) < 50 else (0, 0, 255)
    cv2.line(line_image, (frame_center, height - 20), (int(frame_center + offset), height - 20), offset_color, 4)
    
    offset_text = f"Offset: {abs(offset):.1f}px {'LEFT' if offset < 0 else 'RIGHT' if offset > 0 else 'CENTER'}"
    steering_text = f"Steering: {steering_angle:.1f}°"
    action_text = "CENTERING..." if abs(offset) > 20 else "CENTERED"
    
    cv2.putText(line_image, offset_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(line_image, steering_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(line_image, action_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    
    return result, binary, steering_angle, offset

def main():
    cap = cv2.VideoCapture('lane3.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    cv2.namedWindow('Lane Detection', cv2.WINDOW_NORMAL)
    
    # Initialize steering filter
    steering_filter = SteeringFilter(smooth_factor=0.95)  # Higher value = smoother
    
    # Get original FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate delay for desired playback speed (e.g., 2x slower)
    delay = int((1.0/fps) * 2000)  # in milliseconds
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        processed, binary, steering_angle, offset = process_frame(frame, steering_filter)
        stacked = np.hstack((frame, processed))
        cv2.imshow('Lane Detection', stacked)
        
        # Use waitKey with calculated delay
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
