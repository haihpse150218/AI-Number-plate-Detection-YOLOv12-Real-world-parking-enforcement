import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np
import os
import xlwings as xw
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize PaddleOCR with angle classifier
ocr = PaddleOCR(use_angle_cls=True)

# Initialize video capture and YOLO model
cap = cv2.VideoCapture('demo.mp4')
model = YOLO("best.pt")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = 1020  # Same as resize width
height = 500  # Same as resize height

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output2.mp4', fourcc, fps, (width, height))

# Load class names
with open("coco1.txt", "r") as f:
    class_names = f.read().splitlines()

def save_to_excel(detected_text, filename=None):
    """Save detected text to an Excel file with the current date and time."""
    try:
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d")
        current_time = current_datetime.strftime("%H:%M:%S")

        if filename is None:
            filename = os.path.join(os.getcwd(), f"{current_date}_detected_plates.xlsx")

        app = xw.App(visible=False)
        try:
            if os.path.exists(filename):
                wb = app.books.open(filename)
            else:
                wb = app.books.add()
                wb.save(filename)
        except Exception as e:
            logging.error(f"Failed to open/create workbook: {e}")
            wb = app.books.add()
            wb.save(filename)

        sheet_name = current_date
        if sheet_name in [s.name for s in wb.sheets]:
            sheet = wb.sheets[sheet_name]
        else:
            sheet = wb.sheets.add(sheet_name)

        last_row = sheet.range('A' + str(sheet.cells.last_cell.row)).end('up').row
        next_row = last_row + 1 if last_row > 1 else 1

        sheet.range(f'A{next_row}').value = detected_text
        sheet.range(f'B{next_row}').value = current_date
        sheet.range(f'C{next_row}').value = current_time

        sheet.range('A:C').autofit()
        sheet.range(f'B{next_row}').number_format = 'yyyy-mm-dd'

        wb.save()
        wb.close()
        app.quit()
        logging.info(f"Saved to {filename}, row {next_row}")
    except Exception as e:
        logging.error(f"Error saving to Excel: {e}")
    finally:
        try:
            app.quit()
        except:
            pass

def perform_ocr(image_array):
    """Perform OCR on an image array and return detected text."""
    if image_array is None or image_array.size == 0:
        logging.warning("Empty image provided for OCR")
        return ""

    try:
        results = ocr.ocr(image_array, rec=True)
        detected_text = []
        if results[0] is not None:
            for result in results[0]:
                text = result[1][0]
                detected_text.append(text)
        return ''.join(detected_text)
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""

def RGB(event, x, y, flags, param):
    """Mouse callback to print coordinates."""
    if event == cv2.EVENT_MOUSEMOVE:
        logging.info(f"Mouse position: [{x}, {y}]")

# Set up window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Define detection area
area = [(612, 316), (598, 365), (938, 344), (924, 307)]
counter = []
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        logging.info("End of video stream")
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (width, height))
    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            x1, y1, x2, y2 = box
            class_name = class_names[class_id]

            # Check if the detection is within the area
            result = cv2.pointPolygonTest(np.array(area, np.int32), (x1, y1), False)
            if result >= 0 and track_id not in counter:
                counter.append(track_id)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    logging.warning(f"Empty crop for track ID {track_id}")
                    continue

                crop = cv2.resize(crop, (110, 70))
                text = perform_ocr(crop)
                if text:
                    text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '').replace('-', ' ')
                    save_to_excel(text)
                    logging.info(f"Detected plate: {text}")

                    # Draw bounding box and license plate number
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    
    # Write frame to output video
    output_video.write(frame)
    
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'Esc'
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
logging.info("Script terminated")