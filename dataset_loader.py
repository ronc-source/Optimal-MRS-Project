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
#   text:token_seq -> text (text body of user review / type str)

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
#   description_token_seq -> description (array of sentence descriptions / type list)
#   images <TBD>

#  images (list)


import json
import csv
import os

# Parameters
currentDirectory = "C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project"

amazonReviewFile = currentDirectory + "/dataset/review_Amazon_Fashion.jsonl"
amazonMetaFile = currentDirectory + "/dataset/meta_Amazon_Fashion.jsonl"

outputDir = currentDirectory + "/atomic_data_path/Amazon_Fashion"


'''
# 1. Build .inter file
with open(amazonReviewFile, "r", encoding="utf-8") as fp, open(outputDir + "/Amazon_Fashion.inter", "w", newline = "", encoding="utf-8") as finter:
    writer = csv.writer(finter, delimiter='\t')
    
    writer.writerow([
        "user_id:token",
        "parent_asin:token",
        "rating:float",
        "timestamp:float",
        "text:token_seq"
    ])

    for line in fp:
        dataLine = json.loads(line.strip())

        # Convert text review to token sequence, split with spaces
        textTokenSeq = " ".join(dataLine["text"].split(" "))

        writer.writerow([
            dataLine["user_id"],
            dataLine["parent_asin"],
            dataLine["rating"],
            dataLine["timestamp"],
            textTokenSeq
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

# 3. Build .item file -> same images as URL from jsonl -> create custom model to process these images and convert them to image embeddings uising vit
import requests
from PIL import Image
import torch
from torchvision import transformers
import timm

# Use ViT to create image embeddings

