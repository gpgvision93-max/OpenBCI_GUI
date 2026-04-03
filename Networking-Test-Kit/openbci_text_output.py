import pyopenbci as obci
import numpy as np
import time
from datetime import datetime

class TextOutputToFile:
    def __init__(self, filename="openbci_data.csv"):
        self.filename = filename
        self.file = None
        self.is_recording = False
        
    def handle_sample(self, sample):
        """Write data to text file with timestamp"""
        # Fix: Properly extract data from sample object
        # Different versions of pyopenbci have different sample structures
        try:
            # Try to access sample.data (for older versions)
            data = sample.data
        except AttributeError:
            # For newer versions, data might be directly in sample
            data = sample
            
        # Extract timestamp (assuming sample has timestamp attribute)
        try:
            timestamp = sample.timestamp
        except AttributeError:
            # If no timestamp, use current time
            timestamp = time.time()
        
        # Create text record
        # Ensure data is a list or array before processing
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            data_list = list(data)
        else:
            data_list = [data] if not isinstance(data, list) else data
            
        record = f"{timestamp:.3f}, {', '.join([f'{x:.2f}' for x in data_list])}\n"
        
        # Write to file
        if self.file:
            self.file.write(record)
            self.file.flush()  # Ensure it's written immediately
            
        # Also print to console
        print(f"Data logged: {record.strip()}")
        
    def start_recording(self):
        """Start recording to both file and console"""
        print(f"Starting OpenBCI text output to {self.filename}")
        print("Press Ctrl+C to stop")
        
        try:
            # Open file for writing
            self.file = open(self.filename, 'w')
            self.file.write("Timestamp,Channel1,Channel2,Channel3,Channel4,Channel5,Channel6,Channel7,Channel8\n")
            
            # Start OpenBCI
            self.board = obci.OpenBCIBoard()
            self.board.start_streaming(self.handle_sample)
            self.is_recording = True
            
            while self.is_recording:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Failed to start recording: {e}")
        finally:
            if self.file:
                self.file.close()
                self.file = None
            if hasattr(self, 'board'):
                try:
                    self.board.stop_streaming()
                except Exception:
                    pass
            
    def stop_recording(self):
        """Stop recording and close file"""
        self.is_recording = False
        if hasattr(self, 'board'):
            self.board.stop_streaming()
        if self.file:
            self.file.close()
        print("Recording stopped and file closed")

# For PyCharm run configuration:
if __name__ == "__main__":
    # Run with text output to file
    text_logger = TextOutputToFile("my_openbci_data.csv")
    
    try:
        text_logger.start_recording()
    except KeyboardInterrupt:
        print("\nStopping...")
        text_logger.stop_recording()
