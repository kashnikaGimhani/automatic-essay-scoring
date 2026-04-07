import torch
from transformers import T5ForConditionalGeneration

print("before:", torch.cuda.mem_get_info())

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False

print("after cpu load:", torch.cuda.mem_get_info())

model.to("cuda:0")
print("after model.to:", torch.cuda.mem_get_info())
print("SUCCESS")