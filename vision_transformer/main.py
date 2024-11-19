from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
#model.to(device)
print('Model is loaded into the gpu')

from PIL import Image
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/800px-Cat03.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "[INST] <image>\n Generate me a caption for this image  [/INST]"

inputs = processor(prompt, image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)


print(processor.decode(output[0], skip_special_tokens=True))

