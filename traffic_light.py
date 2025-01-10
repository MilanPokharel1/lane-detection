import torch
import cv2
import numpy as np
from PIL import Image

class TrafficLightDetector:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the traffic light detector
        """
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.conf = confidence_threshold
        self.traffic_light_class_id = 9
        
    def preprocess_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)
    
    def detect_traffic_lights(self, frame):
        processed_frame = self.preprocess_frame(frame)
        results = self.model(processed_frame)
        
        detections = []
        for detection in results.xyxy[0]:
            if int(detection[5]) == self.traffic_light_class_id:
                x1, y1, x2, y2 = map(int, detection[:4])
                confidence = float(detection[4])
                
                traffic_light_roi = frame[y1:y2, x1:x2]
                color = self.determine_light_color(traffic_light_roi)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'color': color
                })
        
        return detections
    
    def determine_light_color(self, roi):
        if roi.size == 0:
            return 'unknown'
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'red': [(0, 70, 50), (10, 255, 255)],
            'yellow': [(20, 70, 50), (35, 255, 255)],
            'green': [(35, 70, 50), (85, 255, 255)]
        }
        
        max_pixels = 0
        dominant_color = 'unknown'
        
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            pixel_count = cv2.countNonZero(mask)
            
            if pixel_count > max_pixels:
                max_pixels = pixel_count
                dominant_color = color
        
        return dominant_color
    
    def process_video_side_by_side(self, video_path):
        """
        Process video and show original and processed frames side by side
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create window
        window_name = 'Traffic Light Detection - Side by Side'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Create a copy for detection
            processed_frame = frame.copy()
            
            # Detect traffic lights
            detections = self.detect_traffic_lights(processed_frame)
            
            # Draw detections
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                color = detection['color']
                confidence = detection['confidence']
                
                # Choose box color based on traffic light color
                box_color = {
                    'red': (0, 0, 255),
                    'yellow': (0, 255, 255),
                    'green': (0, 255, 0),
                    'unknown': (128, 128, 128)
                }.get(color, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw label with background
                label = f"{color}: {confidence:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(processed_frame, (x1, y1-25), (x1 + label_w, y1), box_color, -1)
                cv2.putText(processed_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Create side-by-side view
            combined_frame = np.hstack((frame, processed_frame))
            
            # Add labels for original and processed views
            cv2.putText(combined_frame, "Original", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, "Processed", (frame_width + 10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show combined frame
            cv2.imshow(window_name, combined_frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Add a small delay to make the video playback more manageable
            cv2.waitKey(int(1000/fps))
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    # Initialize detector
    detector = TrafficLightDetector(confidence_threshold=0.5)
    
    # Process video with side-by-side view
    video_path = 'lane_video.mp4'
    detector.process_video_side_by_side(video_path)

if __name__ == '__main__':
    main()