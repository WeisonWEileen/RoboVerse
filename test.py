# import torch
# from transformers import AutoImageProcessor, AutoModel
# from transformers.image_utils import load_image

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = load_image(url)

# pretrained_model_name = "/home/panwei/Downloads/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
# processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(
#     pretrained_model_name,
#     device_map="auto",
# )

# inputs = processor(images=image, return_tensors="pt").to(model.device)
# with torch.inference_mode():
#     outputs = model(**inputs)

# pooled_output = outputs.pooler_output
# print("Pooled output shape:", pooled_output.shape)


from transformers import pipeline
from transformers.image_utils import load_image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction",
)
features = feature_extractor(image)
breakpoint()
print(type(features))