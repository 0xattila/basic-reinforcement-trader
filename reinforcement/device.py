import logging

import torch

logger = logging.getLogger(__name__)

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
    logger.info("CUDA is enabled")
else:
    logger.info("CUDA is disabled")
