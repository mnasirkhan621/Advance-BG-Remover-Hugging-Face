import io
import torch
import numpy as np
import logging
import traceback
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from torchvision import transforms
from safetensors.torch import load_file

from models.birefnet import BiRefNet
from utils import check_state_dict

from fastapi.middleware.cors import CORSMiddleware

# =========================
# Logging Config
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# App
# =========================
app = FastAPI(title="BiRefNet Background Removal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# =========================
# Load model ONCE
# =========================
try:
    logger.info("Loading model architecture...")
    model = BiRefNet(bb_pretrained=False).to(device).eval()
    
    # Check for quantized model first
    import os
    if os.path.exists("weights/quantized_model.pth") and device.type == 'cpu':
        logger.info("Found quantized model 'weights/quantized_model.pth'. Loading...")
        # Apply quantization stub structure
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        state_dict = torch.load("weights/quantized_model.pth", map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Quantized model loaded successfully!")
    else:
        logger.info("Loading original weights...")
        state_dict = load_file("weights/model.safetensors", device=str(device))
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)
        logger.info("Original model loaded successfully!")

except Exception as e:
    logger.error(f"Error loading model: {e}")
    traceback.print_exc()
    # vital error, maybe exit or let it crash on request

# =========================
# Preprocessing
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])

MAX_SIDE = 1536
DIVISOR = 32  # REQUIRED for BiRefNet


def round_to_multiple(x, base=32):
    return int(np.round(x / base) * base)


def smart_resize(img: Image.Image):
    w, h = img.size

    scale = MAX_SIDE / max(w, h)
    if scale > 1:
        scale = 1.0

    new_w = round_to_multiple(w * scale, DIVISOR)
    new_h = round_to_multiple(h * scale, DIVISOR)

    new_w = max(DIVISOR, new_w)
    new_h = max(DIVISOR, new_h)

    return img.resize((new_w, new_h), Image.BICUBIC), (w, h)


# =========================
# API Endpoint
# =========================
@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    try:
        logger.info(f"Received request for file: {file.filename}")
        img = Image.open(file.file).convert("RGB")
        W, H = img.size
        logger.info(f"Original image size: {W}x{H}")

        img_resized, _ = smart_resize(img)
        logger.info(f"Resized image size: {img_resized.size}")

        x = transform(img_resized).unsqueeze(0).to(device)
        logger.info("Input tensor created. Starting inference...")

        with torch.no_grad():
            preds = model(x)
            logger.info("Inference models(x) done.")
            
            # Helper to debug output structure if needed
            if isinstance(preds, (list, tuple)):
                 logger.info(f"Model output type: list/tuple of length {len(preds)}")
            
            mask = torch.sigmoid(preds[-1])[0, 0].cpu().numpy()
            logger.info("Mask generated.")

        # Resize mask back
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        mask = mask.resize((W, H), Image.BICUBIC)
        logger.info("Mask resized to original dimensions.")

        rgba = np.dstack([np.array(img), np.array(mask)])
        out = Image.fromarray(rgba, "RGBA")

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        
        logger.info("Returning response image.")
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": "Internal Server Error", "details": str(e)})

if __name__ == "__main__":
    import uvicorn
    print("Starting server... Access Swagger UI at: http://localhost:8080/docs")
    uvicorn.run(app, host="0.0.0.0", port=8070)
