import torch
import torch.nn as nn
import time
import logging
import numpy as np
from models.birefnet import BiRefNet
from safetensors.torch import load_file
from utils import check_state_dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("quant_test")

def test_quantization():
    device = torch.device("cpu") # Quantization is for CPU
    logger.info(f"Using device: {device}")

    # 1. Load Original Model
    logger.info("Loading original float32 model...")
    model = BiRefNet(bb_pretrained=False).to(device).eval()
    state_dict = load_file("weights/model.safetensors", device=str(device))
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    
    # 2. Create Dummy Input
    x = torch.randn(1, 3, 1024, 1024).to(device)
    
    # 3. Benchmark Original
    logger.info("Warmup original...")
    with torch.no_grad():
        model(x)
        
    logger.info("Benchmarking original model (1 run)...")
    start = time.time()
    with torch.no_grad():
        preds_orig = model(x)
    end = time.time()
    time_orig = end - start
    logger.info(f"Original Time: {time_orig:.2f}s")
    
    # 4. Quantize
    logger.info("Quantizing model (Dynamic INT8)...")
    # Quantize Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    
    # 5. Benchmark Quantized
    logger.info("Warmup quantized...")
    with torch.no_grad():
        quantized_model(x)
        
    logger.info("Benchmarking quantized model (1 run)...")
    start = time.time()
    with torch.no_grad():
        preds_quant = quantized_model(x)
    end = time.time()
    time_quant = end - start
    logger.info(f"Quantized Time: {time_quant:.2f}s")
    
    # 6. Calculate Speedup and Accuracy Drop
    speedup = time_orig / time_quant
    logger.info(f"Speedup: {speedup:.2f}x")
    
    # Compare outputs (last prediction which is the mask)
    out_orig = torch.sigmoid(preds_orig[-1])[0, 0].numpy()
    out_quant = torch.sigmoid(preds_quant[-1])[0, 0].numpy()
    
    mse = np.mean((out_orig - out_quant) ** 2)
    mae = np.mean(np.abs(out_orig - out_quant))
    
    logger.info(f"Mean Squared Error (MSE): {mse:.6f}")
    logger.info(f"Mean Absolute Error (MAE): {mae:.6f}")
    
    if mse < 0.005:
        logger.info("SUCCESS: Accuracy is preserved (Error is very low).")
        # Save quantized model
        # Note: We can't easily save the quantized model structure with safetensors cleanly if it changes class type
        # usually simpler to quantize on load for dynamic. 
        # But we can try torch.save for the whole model or state dict.
        torch.save(quantized_model.state_dict(), "weights/quantized_model.pth")
        logger.info("Saved quantized weights to weights/quantized_model.pth")
    else:
        logger.warning("WARNING: Accuracy drop might be significant!")

if __name__ == "__main__":
    test_quantization()
