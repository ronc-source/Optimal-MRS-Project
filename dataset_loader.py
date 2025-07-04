import json

#currentDirectory = "C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project"

# Test Load Amazon Fashion Reviews

'''
file = currentDirectory + "/dataset/review_Amazon_Fashion.jsonl"

with open(file, 'r') as fp:
    for line in fp:
        print(json.loads(line.strip()))
        break
'''

# Test Load Amazon Fashion Metadata (Text + Images)

'''
file = currentDirectory + "/dataset/meta_Amazon_Fashion.jsonl"

with open(file, 'r') as fp:
    for line in fp:
        print(json.loads(line.strip()))
        break
'''

# Link Amazon Review and Amazon Product Metadata using parent_asin field
# asin ID of the product is the same as parent_asin for user reviews and item metadata


# Atomic File Mappings

##############################################################
# 1. meta_review_Amazon_Fashion.inter (User-item interaction)
##############################################################

# From review dataset:
#   user_id:token -> user_id (str)
#   parent_asin:token -> parent_asin (str)
#   rating:float -> rating (from 1.0 to 5.0 / type float)
#   timestamp:float -> timestamp (int)
#   text_emb:float_seq -> text (BERT text embedding of text body of user review / type str)

####################################################
# 2. meta_review_Amazon_Fashion.user (User feature)
####################################################

# From review dataset:
#   user_id:token -> user_id (str)
# NOTE: No other user related features are provided in the dataset, this atomic file may not be required

####################################################
# 3. meta_review_Amazon_Fashion.item (Item feature)
####################################################

# From meta dataset:
#   parent_asin:token -> parent_asin (str)
#   average_rating:float -> average_rating (float)
#   rating_number:float -> rating_number (int)
#   price:float -> price (float)
#   description:token_seq -> description (array of sentence descriptions / type list)
#   images_url:token -> images (.jpg URL for "large" images / only using 1 image)

#  images (list)


import json
import csv
import torch
from transformers import BertTokenizer, BertModel
import os

# Parameters
currentDirectory = "C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project"

amazonReviewFile = currentDirectory + "/dataset/review_Amazon_Fashion.jsonl"
amazonMetaFile = currentDirectory + "/dataset/meta_Amazon_Fashion.jsonl"

outputDir = currentDirectory + "/atomic_data_path/Amazon_Fashion"

# Max amnount of text tokens we will feed to BERT
MAX_TEXT_TOKEN_LENGTH = 128

# Load BERT

# Use GPU for tensor and model, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# bert-base-uncased is a pretrained English model

# tokenizer stays on CPU
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model is moved to device, so its weights and any new tensors it allocates live on GPU if available
bertModel = BertModel.from_pretrained('bert-base-uncased').to(device)

# disable dropout as it is not needed
bertModel.eval()


# Text Encoder - BERT
def encodeText(text, tokenizer, model, device, maxLength = MAX_TEXT_TOKEN_LENGTH):
    """
    Return a torch.Tensor of shape [D]
    If text is blank/empty, return a zero vector of appropriate size
    """

    # 768 dim embedding
    D = model.config.hidden_size

    # text is empty, so default to zero vector
    if not text.strip():
        return torch.zeros(D, device=device)
    
    inputs = tokenizer(
        text,
        max_length = maxLength,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )

    # Move token tensors to the same device as the model
    inputs = {k : v.to(device) for k, v in inputs.items()}

    # Inference on device
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Return the [CLS] embedding on that device
    return outputs.last_hidden_state[0, 0, :].to(device)


'''
# 1. Build .inter file
with open(amazonReviewFile, "r", encoding="utf-8") as fp, open(outputDir + "/Amazon_Fashion.inter", "w", newline = "", encoding="utf-8") as finter:
    writer = csv.writer(finter, delimiter='\t')
    
    writer.writerow([
        "user_id:token",
        "parent_asin:token",
        "rating:float",
        "timestamp:float",
        "text_emb:float_seq"
    ])

    for line in fp:
        dataLine = json.loads(line.strip())

        # Convert text review to token sequence, split with spaces
        textEmbedding = encodeText(dataLine["text"], bertTokenizer, bertModel, device)

        # print(len(textEmbedding)) -> 768
        
        textEmbeddingSequence = " ".join(f"{x:.6f}" for x in textEmbedding)

        writer.writerow([
            dataLine["user_id"],
            dataLine["parent_asin"],
            dataLine["rating"],
            dataLine["timestamp"],
            textEmbeddingSequence
        ])
'''

'''
# 2. Build .user file
with open(amazonReviewFile, "r", encoding="utf-8") as fp, open(outputDir + "/Amazon_Fashion.user", "w", newline = "", encoding="utf-8") as fuser:
    writer = csv.writer(fuser, delimiter='\t')

    writer.writerow([
        "user_id:token"
    ])

    for line in fp:
        dataLine = json.loads(line.strip())

        writer.writerow([
            dataLine["user_id"]
        ])
'''


# 3. Build .item file -> save images as URL from jsonl -> create custom model to process these images and convert them to image embeddings uising vit
import requests
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50
import io

# Use GPU for tensor and model, otherwise CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Give us ImageNet trained weights so our embeddings will be able to capture rich visual features
model = resnet50(pretrained=True)

# Save feature dimension before dropping the classification head -> value is usually 2048 for resnet
resnetFeatDim = model.fc.in_features

# Remove the final classification layer so the model returns the raw 2048 dimension feature vector instead of class scores
model.fc = torch.nn.Identity()

# turn off dropout since we are only doing inference (using a model to make predictions or generate output based on new, unseen data)
model = model.to(device).eval()


# Setup ImageNet preprocessing
preprocess = transforms.Compose([

    # Resize images to the same dimensions that ResNet was trained on
    transforms.Resize(256),
    transforms.CenterCrop(224),

    # Convert PIL image to tensor multidimensional array
    transforms.ToTensor(),

    # Use the same values that ImageNet models expect so the activation function lands in the expected range
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


with open(amazonMetaFile, "r", encoding="utf-8") as fp, open(outputDir + "/Amazon_Fashion.item", "w", newline = "", encoding="utf-8") as fitem:
    writer = csv.writer(fitem, delimiter='\t')

    writer.writerow([
        "parent_asin:token",
        "average_rating:float",
        "rating_number:float",
        "price:float",
        "description_emb:float_seq",
        "images_emb:float_seq"
    ])

    for line in fp:
        dataLine = json.loads(line.strip())

        # Get description text embedding via BERT

        descTokenSeq = " ".join(dataLine["description"])

        # Convert text review to token sequence, split with spaces
        descEmbedding = encodeText(descTokenSeq, bertTokenizer, bertModel, device)

        # Should be 768
        #print("Description BERT embedding length:", len(descEmbedding))
        
        descEmbeddingSequence = " ".join(f"{x:.6f}" for x in descEmbedding)

        imageURL = ""

        if dataLine["images"]:
            if "large" in dataLine["images"][0]:
                imageURL = dataLine["images"][0]["large"]

        if imageURL:
            try:
                # Download with a 10 second timeout to avoid broken URLs
                resp = requests.get(imageURL, timeout=10)

                # Convert image to RGB as ResNet does not accept grayscale
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")

                # Add batch dimensions - shape [1, 3, 224, 224]
                x = preprocess(img).unsqueeze(0).to(device)

                # So PyTorch does not build a gradient graph and we save memory
                with torch.no_grad():
                    # shape [2048]
                    feat = model(x).squeeze(0)

                # Extract a pure Python list of floats to represent the image embedding
                imageEmb = feat.tolist()
            except Exception:
                # In case of any issues with the image embedding, use a zero vector
                imageEmb = [0.0] * resnetFeatDim
        else:
            imageEmb = [0.0] * resnetFeatDim
        
        # Image embedding dimension length -> 2048
        #print("Image ResNet embedding dimension length", len(imageEmb))

        # Set the image embedding to space separated floats for the float_seq requirement in the atomic file
        # Set the float value to 6 decimal places
        imageEmbeddingSequence = " ".join(f"{v:.6f}" for v in imageEmb)

        writer.writerow([
            dataLine["parent_asin"],
            dataLine["average_rating"],
            dataLine["rating_number"],
            dataLine["price"],
            descEmbeddingSequence,
            imageEmbeddingSequence
        ])