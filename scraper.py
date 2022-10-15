from PIL import Image
import numpy as np
import clip
import requests
import wget
import torch
from torch import autocast
import os
from huggingface_hub import HfApi
from transformers import CLIPTextModel, CLIPTokenizer
# import html
device = "cuda" if torch.cuda.is_available() else "cpu"  # "mps"
clp, preprocess = clip.load("ViT-L/14", device=device)

# from share_btn import community_icon_html, loading_icon_html, share_js

api = HfApi()
models_list = api.list_models(author="sd-concepts-library", sort="likes", direction=-1)
models = []

# pipe = 
# 
#StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", 
# use_auth_token=True, revision="fp16", 
# torch_dtype=torch.float16).to("cuda")

def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  return
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  embeds.to(dtype)

  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  i = 1
  while(num_added_tokens == 0):
    print(f"The tokenizer already contains the token {token}.")
    token = f"{token[:-1]}-{i}>"
    print(f"Attempting to add the token {token}.")
    num_added_tokens = tokenizer.add_tokens(token)
    i+=1
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return token

print("Setting up the public library")
for model in models_list:
  model_content = {}
  model_id = model.modelId
  model_content["id"] = model_id
  embeds_url = f"https://huggingface.co/{model_id}/resolve/main/learned_embeds.bin"
  os.makedirs(model_id,exist_ok = True)
  if not os.path.exists(f"{model_id}/learned_embeds.bin"):
    try:
      wget.download(embeds_url, out=model_id)
    except:
      import traceback
      traceback.print_exc()
      continue
  token_identifier = f"https://huggingface.co/{model_id}/raw/main/token_identifier.txt"
  response = requests.get(token_identifier)
  token_name = response.text
  
  concept_type = f"https://huggingface.co/{model_id}/raw/main/type_of_concept.txt"
  response = requests.get(concept_type)
  concept_name = response.text
  model_content["concept_type"] = concept_name
  images = []
  for i in range(4):
    url = f"https://huggingface.co/{model_id}/resolve/main/concept_images/{i}.jpeg"
    if os.path.exists(f"{model_id}/{i}.npy"):
      continue
    image_download = requests.get(url)
    url_code = image_download.status_code
    if(url_code == 200):
      file = open(f"{model_id}/{i}.jpeg", "wb") ## Creates the file for image
      file.write(image_download.content) ## Saves file content
      file.close()
      images.append(f"{model_id}/{i}.jpeg")
  # model_content["images"] = images
      image = preprocess(Image.open(images[-1])).unsqueeze(0).to(device)
      with torch.no_grad():
        image_features = clp.encode_image(image).detach().cpu().numpy()
      np.save(f"{model_id}/{i}", image_features)

  # learned_token = load_learned_embed_in_clip(f"{model_id}/learned_embeds.bin", pipe.text_encoder, pipe.tokenizer, token_name)
  # model_content["token"] = learned_token
  # models.append(model_content)
