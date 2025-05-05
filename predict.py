from cog import BasePredictor, Input
from llama_cpp import Llama
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def setup(self):
        model_path = "meta-llama-3.1-8b-instruct-abliterated.Q8_0.gguf"
        logger.info(f"Loading model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
            verbose=False
        )
        logger.info("Model loaded successfully")

    def predict(
        self,
        prompt: str = Input(description="Input prompt to generate text from."),
        seed: int = Input(description="Random seed for reproducibility", default=42),
    ) -> str:
        try:
            output = self.llm(
                prompt=prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                seed=seed,
                echo=False
            )
            return output["choices"][0]["text"]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError("Prediction failed.")