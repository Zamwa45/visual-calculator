"""
Hand-Tracking Calculator Application

A computer vision-based calculator that uses hand gestures for input.
Users can perform calculations by pointing at buttons with their index finger.

Dependencies:
    - opencv-python (cv2)
    - cvzone

Author: [Your Name]
Date: [Current Date]
Version: 1.0
"""

import cv2
import logging
from typing import List, Tuple, Optional
from cvzone.HandTrackingModule import HandDetector


class CalculatorButton:
    """
    Represents a calculator button with position, dimensions, and interaction capabilities.
    
    Attributes:
        position (Tuple[int, int]): Top-left corner coordinates (x, y)
        width (int): Button width in pixels
        height (int): Button height in pixels  
        value (str): Display value/symbol for the button
        
    Constants:
        BUTTON_COLOR: Default button background color (light gray)
        BORDER_COLOR: Button border color (dark gray)
        TEXT_COLOR: Button text color (dark gray)
        ACTIVE_COLOR: Button color when pressed (white)
        ACTIVE_TEXT_COLOR: Text color when button is pressed (black)
    """
    
    # Color constants
    BUTTON_COLOR = (225, 225, 225)
    BORDER_COLOR = (50, 50, 50)
    TEXT_COLOR = (50, 50, 50)
    ACTIVE_COLOR = (255, 255, 255)
    ACTIVE_TEXT_COLOR = (0, 0, 0)
    
    def __init__(self, position: Tuple[int, int], width: int, height: int, value: str):
        """
        Initialize a calculator button.
        
        Args:
            position: Top-left corner coordinates (x, y)
            width: Button width in pixels
            height: Button height in pixels
            value: Display value/symbol for the button
        """
        self.position = position
        self.width = width
        self.height = height
        self.value = value

    def draw(self, img: cv2.Mat, is_active: bool = False) -> None:
        """
        Draw the button on the given image.
        
        Args:
            img: OpenCV image array to draw on
            is_active: Whether to draw button in active/pressed state
        """
        x, y = self.position
        end_pos = (x + self.width, y + self.height)
        
        # Choose colors based on button state
        bg_color = self.ACTIVE_COLOR if is_active else self.BUTTON_COLOR
        text_color = self.ACTIVE_TEXT_COLOR if is_active else self.TEXT_COLOR
        
        # Draw button background
        cv2.rectangle(img, self.position, end_pos, bg_color, cv2.FILLED)
        
        # Draw button border
        cv2.rectangle(img, self.position, end_pos, self.BORDER_COLOR, 3)
        
        # Calculate text position for centering
        text_offset_x = 30 if not is_active else 25
        text_offset_y = 70 if not is_active else 80
        text_scale = 2 if not is_active else 5
        text_thickness = 2 if not is_active else 5
        
        text_pos = (x + text_offset_x, y + text_offset_y)
        
        # Draw button text
        cv2.putText(img, self.value, text_pos, cv2.FONT_HERSHEY_PLAIN,
                   text_scale, text_color, text_thickness)

    def is_clicked(self, x: int, y: int) -> bool:
        """
        Check if the given coordinates are within the button boundaries.
        
        Args:
            x: X coordinate to check
            y: Y coordinate to check
            
        Returns:
            bool: True if coordinates are within button, False otherwise
        """
        return (self.position[0] < x < self.position[0] + self.width and
                self.position[1] < y < self.position[1] + self.height)


class HandTrackingCalculator:
    """
    A hand-tracking calculator application using computer vision.
    
    This class manages the calculator interface, hand detection, and calculation logic.
    Users can interact with the calculator by pointing their index finger at buttons.
    
    Attributes:
        BUTTON_LAYOUT: 2D array defining calculator button layout
        DISPLAY_AREA: Coordinates for the equation display area
        CLICK_THRESHOLD: Distance threshold for registering finger clicks
        DELAY_FRAMES: Number of frames to wait between button clicks
    """
    
    # Calculator layout configuration
    BUTTON_LAYOUT = [
        ['7', '8', '9', '*'],
        ['4', '5', '6', '-'], 
        ['1', '2', '3', '+'],
        ['0', '/', '.', '=']
    ]
    
    # UI configuration constants
    DISPLAY_AREA = (800, 70, 400, 100)  # x, y, width, height
    BUTTON_START_POS = (800, 150)
    BUTTON_SIZE = (100, 100)
    CLICK_THRESHOLD = 50
    DELAY_FRAMES = 10
    
    def __init__(self, camera_index: int = 0, detection_confidence: float = 0.8):
        """
        Initialize the hand-tracking calculator.
        
        Args:
            camera_index: Index of camera to use (default: 0)
            detection_confidence: Hand detection confidence threshold (0.0-1.0)
        """
        self.equation = ""
        self.delay_counter = 0
        self.buttons = self._create_buttons()
        self.clicked_button_index = -1
        
        # Initialize camera and hand detector
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera with index {camera_index}")
            
        self.detector = HandDetector(detectionCon=detection_confidence, maxHands=1)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _create_buttons(self) -> List[CalculatorButton]:
        """
        Create calculator button objects based on the defined layout.
        
        Returns:
            List[CalculatorButton]: List of button objects
        """
        buttons = []
        start_x, start_y = self.BUTTON_START_POS
        button_width, button_height = self.BUTTON_SIZE
        
        for row in range(len(self.BUTTON_LAYOUT)):
            for col in range(len(self.BUTTON_LAYOUT[row])):
                x_pos = col * button_width + start_x
                y_pos = row * button_height + start_y
                value = self.BUTTON_LAYOUT[row][col]
                
                button = CalculatorButton((x_pos, y_pos), button_width, button_height, value)
                buttons.append(button)
                
        return buttons
        
    def _draw_display(self, img: cv2.Mat) -> None:
        """
        Draw the equation display area and current equation.
        
        Args:
            img: OpenCV image array to draw on
        """
        x, y, width, height = self.DISPLAY_AREA
        
        # Draw display background
        cv2.rectangle(img, (x, y), (x + width, y + height),
                     CalculatorButton.BUTTON_COLOR, cv2.FILLED)
        
        # Draw display border  
        cv2.rectangle(img, (x, y), (x + width, y + height),
                     CalculatorButton.BORDER_COLOR, 3)
        
        # Draw equation text
        text_pos = (x + 10, y + 60)
        cv2.putText(img, self.equation, text_pos, cv2.FONT_HERSHEY_PLAIN,
                   3, CalculatorButton.TEXT_COLOR, 3)
    
    def _process_button_click(self, button_index: int) -> None:
        """
        Process a button click and update the equation accordingly.
        
        Args:
            button_index: Index of the clicked button in the buttons list
        """
        button_value = self.buttons[button_index].value
        
        try:
            if button_value == '=':
                if self.equation:
                    # Safely evaluate the equation
                    result = eval(self.equation)
                    self.equation = str(result)
                    self.logger.info(f"Calculated result: {result}")
            else:
                self.equation += button_value
                self.logger.info(f"Added '{button_value}' to equation")
                
        except (SyntaxError, ZeroDivisionError, ValueError) as e:
            self.logger.error(f"Calculation error: {e}")
            self.equation = "Error"
    
    def _handle_hand_interaction(self, hands: List, img: cv2.Mat) -> None:
        """
        Process hand detection and handle button interactions.
        
        Args:
            hands: List of detected hands from hand detector
            img: OpenCV image array
        """
        if not hands:
            return
            
        # Get hand landmarks
        hand = hands[0]
        landmarks = hand['lmList']
        
        # Calculate distance between index finger tip and middle finger tip
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        
        distance, _, img = self.detector.findDistance(index_tip, middle_tip, img)
        finger_x, finger_y = index_tip
        
        # Check for button click (fingers close together indicates pointing gesture)
        if distance < self.CLICK_THRESHOLD and self.delay_counter == 0:
            # Check which button was clicked
            for i, button in enumerate(self.buttons):
                if button.is_clicked(finger_x, finger_y):
                    self.clicked_button_index = i
                    self._process_button_click(i)
                    self.delay_counter = 1
                    break
    
    def _update_delay_counter(self) -> None:
        """Update the delay counter to prevent multiple rapid clicks."""
        if self.delay_counter > 0:
            self.delay_counter += 1
            if self.delay_counter > self.DELAY_FRAMES:
                self.delay_counter = 0
                self.clicked_button_index = -1
    
    def _draw_buttons(self, img: cv2.Mat) -> None:
        """
        Draw all calculator buttons on the image.
        
        Args:
            img: OpenCV image array to draw on
        """
        for i, button in enumerate(self.buttons):
            is_active = (i == self.clicked_button_index)
            button.draw(img, is_active)
    
    def run(self) -> None:
        """
        Main application loop for the hand-tracking calculator.
        
        Controls:
        - Point index finger at buttons to click them
        - Press 'c' key to clear the equation
        - Press 'q' or ESC key to quit the application
        """
        self.logger.info("Starting Hand-Tracking Calculator...")
        self.logger.info("Controls: Point to click buttons, 'c' to clear, 'q' or ESC to quit")
        
        try:
            while True:
                # Capture frame from camera
                success, img = self.cap.read()
                if not success:
                    self.logger.error("Failed to read from camera")
                    break
                
                # Flip image horizontally for mirror effect
                img = cv2.flip(img, 1)
                
                # Detect hands
                hands, img = self.detector.findHands(img)
                
                # Draw UI elements
                self._draw_display(img)
                self._draw_buttons(img)
                
                # Handle hand interactions
                self._handle_hand_interaction(hands, img)
                
                # Update delay counter
                self._update_delay_counter()
                
                # Display the frame
                cv2.imshow("Hand-Tracking Calculator", img)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    self.equation = ""
                    self.logger.info("Equation cleared")
                elif key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources and close windows."""
        self.logger.info("Cleaning up resources...")
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point for the application."""
    try:
        calculator = HandTrackingCalculator()
        calculator.run()
    except Exception as e:
        print(f"Failed to start calculator: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
