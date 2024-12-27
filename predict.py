from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from threading import Thread
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2VLForConditionalGeneration, 
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/unsloth/QVQ-72B-Preview-bnb-4bit/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        # Load model with memory optimizations
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_CACHE,
            quantization_config=bnb_config,
            device_map="cuda",
            torch_dtype=torch.float16,
            offload_folder="offload"
        )
        
        # Load processor with default token ranges
        self.processor = AutoProcessor.from_pretrained(
            MODEL_CACHE,
        )

    def predict(
        self,
        image: Path = Input(description="Input image file"),
        prompt: str = Input(
            description="Text prompt to guide the model's analysis",
            default="What do you see in this image?"
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=8192,
            ge=1,
            le=8192
        ),
    ) -> str:
        """Run a single prediction on the model"""
        try:
            # Load and process the image
            if not os.path.exists(image):
                raise ValueError(f"Image file not found: {image}")
            
            # Open the image using PIL
            pil_image = Image.open(image)
            
            # Convert RGBA to RGB if necessary
            if pil_image.mode == 'RGBA':
                pil_image = pil_image.convert('RGB')
            
            # Prepare messages with the processed image
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image,  # Pass PIL Image directly
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            # Process the conversation
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process vision information
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Prepare inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to GPU
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v 
                     for k, v in inputs.items()}

            # Generate output with proper error handling
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Process generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            # Decode the output
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            return output_text[0].strip()
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise e