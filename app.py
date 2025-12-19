import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from torchvision import transforms
from safetensors.torch import load_file

from models.birefnet import BiRefNet
from utils import check_state_dict

from fastapi.middleware.cors import CORSMiddleware

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

# =========================
# Load model ONCE
# =========================
model = BiRefNet(bb_pretrained=False).to(device).eval()

state_dict = load_file("weights/model.safetensors", device=str(device))
state_dict = check_state_dict(state_dict)
model.load_state_dict(state_dict)

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

import numpy as np

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
    img = Image.open(file.file).convert("RGB")
    W, H = img.size

    img_resized, _ = smart_resize(img)

    x = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(x)
        mask = torch.sigmoid(preds[-1])[0, 0].cpu().numpy()

    # Resize mask back
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.resize((W, H), Image.BICUBIC)

    rgba = np.dstack([np.array(img), np.array(mask)])
    out = Image.fromarray(rgba, "RGBA")

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    print("Starting server... Access Swagger UI at: http://localhost:8080/docs")
    uvicorn.run(app, host="0.0.0.0", port=8070)
