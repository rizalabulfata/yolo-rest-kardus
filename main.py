import os
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
from typing import Optional, List, Dict, Any, Union
import tempfile
import ultralytics

app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="REST API for object detection using YOLOv8",
    version="1.0.0"
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure these settings based on your environment
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MODEL_PATH = 'best.pt'  # Path to your trained YOLOv8 model

# Load the YOLOv8 model
model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ImageBase64(BaseModel):
    image: str
    conf: Optional[float] = 0.25

class DetectionResponse(BaseModel):
    success: bool
    predictions: List[Dict[str, Any]]
    model: str
    image_shape: Dict[str, int]
    count: int

@app.get("/health")
async def health_check():
    """Endpoint to check if the API is running"""
    return {"status": "healthy", "model": os.path.basename(MODEL_PATH)}

@app.post("/predict", response_model=DetectionResponse)
async def predict(
    request: Request,
    image: Optional[UploadFile] = File(None),
    # conf: Optional[float] = Form(0.25)
):
    """
    Endpoint to make predictions using the YOLOv8 model.
    
    - **image**: Upload an image file
    - **conf**: Confidence threshold (0-1)
    
    Returns JSON with detection results.
    """
    try:
        img = None
        
        # Check if the request contains a file upload
        if image is not None:
            if not allowed_file(image.filename):
                raise HTTPException(status_code=400, detail="Invalid file format. Allowed formats: png, jpg, jpeg, webp")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                contents = await image.read()
                temp.write(contents)
                temp_path = temp.name
            
            # Read the image
            img = cv2.imread(temp_path)
            
            # Delete the temporary file
            os.unlink(temp_path)
        
        # If no file was uploaded, check if the request is JSON with base64 image
        else:
            try:
                # Parse request as JSON
                data = await request.json()
                if 'image' in data:
                    base64_img = data['image']
                    
                    # Update conf if provided in JSON
                    if 'conf' in data:
                        conf = float(data['conf'])
                    
                    # Decode the base64 image
                    try:
                        img_data = base64.b64decode(base64_img)
                        img = Image.open(io.BytesIO(img_data))
                        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail="No 'image' field in JSON")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON")
        
        if img is None:
            raise HTTPException(
                status_code=400, 
                detail="No image provided. Send a file in 'image' field or JSON with base64-encoded 'image'"
            )
        
        # Run YOLOv8 inference
        results = model(img)
        
        # Process the results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2)
                    }
                }
                detections.append(detection)
        
        # Return the predictions
        return {
            "success": True,
            "predictions": detections,
            "model": os.path.basename(MODEL_PATH),
            "image_shape": {"height": img.shape[0], "width": img.shape[1]},
            "count": len(detections)
        }
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# For direct API documentation
@app.get("/")
async def root():
    """Redirect to API documentation"""
    return {
        "message": "YOLOv8 Object Detection API. Visit /docs for API documentation",
        }

# Main entry point when running as a script
# if __name__ == "__main__":
#     # For development:
#     # uvicorn.run(app, host="0.0.0.0", port=8000)
    
#     # For production with FastCGI:
#     from uvicorn.workers import UvicornWorker
#     from asgi_wsgi import WSGIMiddleware
#     from flup.server.fcgi import WSGIServer
    
#     # Convert ASGI to WSGI for FastCGI
#     asgi_app = app
#     wsgi_app = WSGIMiddleware(asgi_app)
    
#     # Run with FastCGI
#     WSGIServer(wsgi_app).run()
