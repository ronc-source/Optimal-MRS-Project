# create atomic files required to train recommender model based on Amazon Fashion dataset from https://amazon-reviews-2023.github.io/

# NOTE:
#   link Amazon Review and Amazon Product Metadata using parent_asin field
#   asin of the product is the same as parent_asin for user reviews and item metadata

# atomic File Mappings

##############################################################
# 1. Amazon_Fashion.inter (User-item interaction)
##############################################################

# from review dataset:
#   user_id:token -> user_id (str)
#   parent_asin:token -> parent_asin (str)
#   rating:float -> rating (from 1.0 to 5.0 / type float)
#   timestamp:float -> timestamp (int)
#   text_emb:float_seq -> text (BERT text embedding of user review / type str)

####################################################
# 2. Amazon_Fashion.user (User feature)
####################################################

# from review dataset:
#   user_id:token -> user_id (str)
# NOTE: No other user related features are provided in the dataset, this atomic file may not be required

####################################################
# 3. Amazon_Fashion.item (Item feature)
####################################################

# from meta dataset:
#   parent_asin:token -> parent_asin (str)
#   average_rating:float -> average_rating (float)
#   rating_number:float -> rating_number (int)
#   price:float -> price (float)
#   description_emb:float_seq -> description (array of sentence descriptions / type list)
#   images_emb:float_seq -> images (.jpg URL for "large" images / only using 1 image)

import json
import csv
import torch
import requests
import io

from PIL import Image
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# constants
CURRENT_DIR = "C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project"
OUTPUT_DIR = CURRENT_DIR + "/atomic_data_path/Amazon_Fashion/"

# max amount of text tokens we will feed to BERT
MAX_TEXT_TOKEN_LENGTH = 128

# tensors are a datatype in PyTorch used to represent information such as weights and inputs
# try to use GPU for PyTorch tensors and models, otherwise use CPU

# CUDA Version -> 12.9 from nvidia-smi -> using 12.8 for torch
# torch = 2.7.1+cu128
# torchvision = 0.22.1+cu128

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# bert-base-uncased is a pretrained English model
bertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# bert model is moved to device, so its weights and any new tensors now live on GPU if available or CPU otherwise
bertModel = BertModel.from_pretrained('bert-base-uncased').to(device)

# set model to evaluation mode and disable dropout to prevent noise being added to embeddings when our multimodal recommender uses this data
bertModel.eval()


# return 768 dimension text embedding of float numbers using BERT
# if text is empty, return a zero vector of size 768
def getTextEmb(device, tokenizer, model, text, maxLength = MAX_TEXT_TOKEN_LENGTH):

    text = text.strip()

    # text is empty, return zero vector of same expected dimension
    if not text:

        # model.config.hidden_size is the 768 dimension embedding
        zeroTensor = torch.zeros(model.config.hidden_size, device=device)

        zeroRes = []
        for i in zeroTensor:

            # convert zero to 6 decimal places
            zeroRes.append(f"{i:.6f}")
        
        # separate embedding values with white space to conform to recbole atomic file expectations of float sequences
        return " ".join(zeroRes)
    
    # prepare a tensor of input tokens for BERT model
    formatInput = tokenizer(
        text,
        max_length = maxLength,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )

    # Move token tensors to the same device as the model (GPU -- if not available then CPU)
    inputDict = {}

    for key, val in formatInput.items():
        inputDict[key] = val.to(device)

    finalInput = {k : v.to(device) for k, v in formatInput.items()}

    # disable gradient tracking in model as it is not required since we are directly generating text embeddings for our recommender
    with torch.no_grad():

        # return a tensor of shape (batchSize, seqLen, hiddenSize)
        res = model(**finalInput)
    
    # get the CLS 768 dimension text embedding and put it on our device
    textEmb = res.last_hidden_state[0, 0, :].to(device)

    textEmbRes = []
    for i in textEmb:

        # format text embedding values to 6 decimal places
        textEmbRes.append(f"{i:.6f}")
    
    # return 768 dim text embedding as a string of values separated by white space to conform to recbole atomic file expectations of float sequence
    return " ".join(textEmbRes)


# build the .inter atomic file
def buildInterFile(datasetFile, outputFileName, outputDir=OUTPUT_DIR):

    # open the dataset file and create + open the output file in the directory specified
    with open(datasetFile, "r", encoding="utf-8") as fp, open(outputDir + outputFileName, "w", newline = "", encoding="utf-8") as finter:

        # set the separator between values as tab
        writer = csv.writer(finter, delimiter='\t')

        # write the header row of the atomic file
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
            textEmbedding = getTextEmb(device, bertTokenizer, bertModel, dataLine["text"])

            writer.writerow([
                dataLine["user_id"],
                dataLine["parent_asin"],
                dataLine["rating"],
                dataLine["timestamp"],
                textEmbedding
            ])


# build the .user atomic file
def buildUserFile(datasetFile, outputFileName, outputDir=OUTPUT_DIR):

    with open(datasetFile, "r", encoding="utf-8") as fp, open(outputDir + outputFileName, "w", newline = "", encoding="utf-8") as fuser:

        writer = csv.writer(fuser, delimiter='\t')

        writer.writerow([
            "user_id:token"
        ])

        for line in fp: 
            dataLine = json.loads(line.strip())

            writer.writerow([
                dataLine["user_id"]
            ])


# setup process to get image embeddings

# use resnet50 model whose weights are pretrained on the ImageNet database
res50Model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# get the number of dimensions for the image feature embedding before the classification is done, this will usually be 2048 for resnet
res50EmbDim = res50Model.fc.in_features

# remove the classification layer at the end of the model so we get the raw 2048 dimension embedding vector instead of a class score
res50Model.fc = torch.nn.Identity()

# move all the model components such as the weights and parameters to the device (preferably GPU, use CPU if this fails) and setup model in evaluation mode (turn off dropout)
# turn off dropout since we are only doing inference (using a model to make predictions or generate output based on new, unseen data)
res50Model = res50Model.to(device).eval()

# setup image preprocessing before feeding image into resnet 50 model - code below produces a 3 x 224 x 224 tensor
# image preprocessing outline and values used, such as for mean and std are inspired from https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.quantization.resnet18.html?highlight=transforms+normalize#:~:text=weights='IMAGENET1K_FBGEMM_V1'%20.-,ResNet18_QuantizedWeights.,DEFAULT%20.&text=The%20inference%20transforms%20are%20available,0.229%2C%200.224%2C%200.225%5D%20.
# resize -> centercrop -> rescale -> normalize using mean and std
preprocess = transforms.Compose([

    # resize and crop out the center of the image to the same dimensions that the resnet model was trained on
    transforms.Resize(256),
    transforms.CenterCrop(224),

    # convert our PIL image to a tensor multidimensional array and rescale values to [0.0, 1.0] before normalization (as expected from docs)
    transforms.ToTensor(),

    # normalize image tensor using the same values as expected from images that resnet trained on from ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# build .item file
#   save images as URL from jsonl
#   preprocess these images
#   convert them to image embeddings using resnet 50

def buildItemFile(datasetFile, outputFileName, outputDir=OUTPUT_DIR):
    with open(datasetFile, "r", encoding="utf-8") as fp, open(outputDir + outputFileName, "w", newline = "", encoding="utf-8") as fitem:
        writer = csv.writer(fitem, delimiter='\t')

        writer.writerow([
            "parent_asin:token",
            "average_rating:float",
            "rating_number:float",
            "price:float",
            "description_emb:float_seq",
            "images_emb:float_seq"
        ])

        resultCounter = 1

        for line in fp:
            dataLine = json.loads(line.strip())

            # description is an array of different text such as Feature and Package Including - join them into one long string for processing
            descText = " ".join(dataLine["description"])

            # get description text embedding using BERT
            descEmb = getTextEmb(device, bertTokenizer, bertModel, descText)

            imgURL = ""

            # try to get the first large main image representing the item
            if dataLine["images"]:
                for img in dataLine["images"]:
                    if img.get("variant") == "MAIN" and img.get("large"):
                        imgURL = img.get("large")
                        break

            if imgURL:
                try:
                    # download with a 10 second timeout to avoid broken URLs
                    imgReq = requests.get(imgURL, timeout=10)

                    # convert PIL image from bytes to an RGB image as resnet50 does not accept grayscale and requires 3 channels (RGB)
                    imgFound = Image.open(io.BytesIO(imgReq.content)).convert("RGB")

                    # preprocess image found from dataset
                    # add a batch dimension to get tensor shape as [1, 3, 224, 224] from [3, 224, 224] for resnet model 
                    # move result tensor to GPU if available, CPU otherwise
                    imgInput = preprocess(imgFound).unsqueeze(0).to(device)

                    # we do not need to compute or track gradients as we are just calling the model to produce embeddings
                    with torch.no_grad():
                        # get model embedding and remove front batch dimension to go from [1, 2048] to [2048] shape using squeeze
                        # we expect a 2048 dimension vector embedding from resnet50
                        imgEmb = res50Model(imgInput).squeeze(0)

                    # convert tensor to a list of float values
                    imgEmb = imgEmb.tolist()

                except Exception:
                    # in case of any issues with the image embedding, use a zero vector
                    imgEmb = [0.0] * res50EmbDim
            else:
                imgEmb = [0.0] * res50EmbDim

            # set the image embedding to space separated floats for the float_seq requirement in the atomic file
            imgEmbRes = []

            for i in imgEmb:
                # set the float value to 6 decimal places
                imgEmbRes.append(f"{i:.6f}")
            
            imgEmbRes = " ".join(imgEmbRes)

            writer.writerow([
                dataLine["parent_asin"],
                dataLine["average_rating"],
                dataLine["rating_number"],
                dataLine["price"],
                descEmb,
                imgEmbRes
            ])
            
            resultCounter += 1
            print("Data line", resultCounter, "has been added to the file:", outputFileName)


if __name__ == "__main__":
    amazonReviewFile = CURRENT_DIR + "/dataset/review_Amazon_Fashion.jsonl"
    amazonMetaFile = CURRENT_DIR + "/dataset/meta_Amazon_Fashion.jsonl"

    #buildInterFile(amazonReviewFile, "Amazon_Fashion.inter")

    #buildUserFile(amazonReviewFile, "Amazon_Fashion.user")

    buildItemFile(amazonMetaFile, "Amazon_Fashion.item")