import torch
import torch.nn as nn
import logging
from models.birefnet import BiRefNet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("verify_quant")

def verify():
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    try:
        # 1. Instantiate Model Structure
        logger.info("Instantiating base model...")
        model = BiRefNet(bb_pretrained=False).to(device).eval()
        
        # 2. Apply Dynamic Quantization Structure 
        # (Must match how it was saved: just nn.Linear quantized)
        logger.info("Applying quantization stub...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        
        # 3. Load State Dict
        logger.info("Loading quantized weights from 'weights/quantized_model.pth'...")
        state_dict = torch.load("weights/quantized_model.pth", map_location=device)
        quantized_model.load_state_dict(state_dict)
        logger.info("Weights loaded successfully.")
        
        # 4. Test Inference
        x = torch.randn(1, 3, 1024, 1024).to(device)
        logger.info("Running inference...")
        with torch.no_grad():
            preds = quantized_model(x)
        logger.info("Inference successful!")
        
    except Exception as e:
        logger.error(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
