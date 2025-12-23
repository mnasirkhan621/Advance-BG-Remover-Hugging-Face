import torch
import logging
from models.birefnet import BiRefNet
from safetensors.torch import load_file
from utils import check_state_dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_model")

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        logger.info("Loading model...")
        model = BiRefNet(bb_pretrained=False).to(device).eval()
        
        state_dict = load_file("weights/model.safetensors", device=str(device))
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)
        logger.info("Model loaded.")

        # Create dummy input
        # BiRefNet expects specific shape? 
        # app.py uses: x = transform(img_resized).unsqueeze(0).to(device)
        # transform normalizes. 
        # Let's just create a random tensor of appropriate shape (1, 3, 1024, 1024)
        x = torch.randn(1, 3, 1024, 1024).to(device)
        logger.info(f"Input shape: {x.shape}")

        logger.info("Starting inference...")
        with torch.no_grad():
            preds = model(x)
        logger.info("Inference done.")
        
        if isinstance(preds, (list, tuple)):
            logger.info(f"Output type: {type(preds)} of len {len(preds)}")
        else:
             logger.info(f"Output type: {type(preds)}")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
