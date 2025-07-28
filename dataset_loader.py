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

import encoders

import time
from datetime import datetime
import logging

from PIL import Image
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, AlbertModel, AlbertTokenizer, AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights

# constants
CURRENT_DIR = "C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project"

LOGGER = logging.getLogger()
logging.basicConfig(filename= CURRENT_DIR + "/modality-encoder-experiment-log/" + "modality-encoder-experiments-" + str(datetime.now()).replace(" ", "-").replace(".", "-").replace(":", "-") + ".log", 
                    encoding="utf-8", level=logging.INFO)


# build the .inter atomic file
def buildInterFile(device, datasetFile, outputFileName, outputDir, textEncoderFunc, textModel, textTokenizer, textEmbColName):

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
            textEmbColName + ":float_seq"
        ])

        resultCounter = 1
        totalTextEmbLatencyTime = 0
        highestGPUMem = 0

        for line in fp:
            dataLine = json.loads(line.strip())

            # start time counter for text embedding inference
            startTextEmbTime = time.perf_counter()

            # Convert text review to token sequence, split with spaces
            textEmbedding = textEncoderFunc(dataLine["text"], device, textModel, textTokenizer)

            # capture end time counter of text embedding inference
            endTextEmbTime =  time.perf_counter()

            # get the time spent to get text embedding
            totalTextEmbLatencyTime += (endTextEmbTime - startTextEmbTime)

            # record the highest GPU memory used so far (in GB) if found (convert from bytes to GB)
            highestGPUMem = max(highestGPUMem, (torch.cuda.max_memory_allocated(device) / (1024 ** 3)))

            writer.writerow([
                dataLine["user_id"],
                dataLine["parent_asin"],
                dataLine["rating"],
                dataLine["timestamp"],
                textEmbedding
            ])

            resultCounter += 1
            print("Data line", resultCounter, "has been added to the file:", outputFileName)

            # NOTE: hard code stop after 200,000 data lines have been added to the file
            if resultCounter >= 200001:
                break
        
        LOGGER.info("#################################################################")
        LOGGER.info(outputFileName + " Embedding Statistics:")
        LOGGER.info("#################################################################\n")
        
        # -1 result counter to not count heading in result and convert to ms from seconds
        avgTextLatency = (totalTextEmbLatencyTime / (resultCounter - 1)) * 1000
        LOGGER.info(f"Average Text Embedding Latency Spent Per Item (ms): {avgTextLatency:.6f}")

        # get total time spent to compute all embeddings (in minutes)
        LOGGER.info(f"Total Time Spent to Perform All Text Embeddings in Dataset (minutes): {(totalTextEmbLatencyTime / 60):.6f}")

        # get the highest GPU memory recorded during the text embedding process
        LOGGER.info(f"Highest GPU Memory Recorded During Embedding Process (GB): {highestGPUMem:.6f}\n")


# build the .user atomic file
def buildUserFile(datasetFile, outputFileName, outputDir):

    with open(datasetFile, "r", encoding="utf-8") as fp, open(outputDir + outputFileName, "w", newline = "", encoding="utf-8") as fuser:

        writer = csv.writer(fuser, delimiter='\t')

        writer.writerow([
            "user_id:token"
        ])

        resultCounter = 1

        for line in fp: 
            dataLine = json.loads(line.strip())

            writer.writerow([
                dataLine["user_id"]
            ])

            resultCounter += 1
            print("Data line", resultCounter, "has been added to the file:", outputFileName)


# build .item file
#   save images as URL from jsonl
#   preprocess these images
#   convert them to image embeddings using resnet 50

def buildItemFile(device, datasetFile, outputFileName, outputDir, textEncoderFunc, textModel, textTokenizer, textEmbColName,
                  imgEncoderFunc, imgModel, imgEmbDim, imgPreProcessor, imgEmbColName, isVit):
    
    with open(datasetFile, "r", encoding="utf-8") as fp, open(outputDir + outputFileName, "w", newline = "", encoding="utf-8") as fitem:
        writer = csv.writer(fitem, delimiter='\t')

        writer.writerow([
            "parent_asin:token",
            "average_rating:float",
            "rating_number:float",
            "price:float",
            textEmbColName + ":float_seq",
            imgEmbColName + ":float_seq"
        ])

        resultCounter = 1
        totalTextEmbLatencyTime = 0
        totalImgEmbLatencyTime = 0
        highestGPUMem = 0

        for line in fp:
            dataLine = json.loads(line.strip())

            # description is an array of different text such as Feature and Package Including - join them into one long string for processing
            descText = " ".join(dataLine["description"])

            startTextEmbTime = time.perf_counter()

            # get description text embedding using text encoder
            descEmb = textEncoderFunc(descText, device, textModel, textTokenizer)

            endTextEmbTime =  time.perf_counter()

            totalTextEmbLatencyTime += (endTextEmbTime - startTextEmbTime)

            imgURL = ""

            # try to get the first large main image representing the item
            if dataLine["images"]:
                for img in dataLine["images"]:
                    if img.get("variant") == "MAIN" and img.get("large"):
                        imgURL = img.get("large")
                        break
            
            startImageEmbTime = time.perf_counter()

            # if the image encoder is ViT
            if isVit:
                imgEmb = imgEncoderFunc(imgURL, device, imgModel, imgPreProcessor)
            else:
                imgEmb = imgEncoderFunc(imgURL, device, imgModel, imgEmbDim, imgPreProcessor)
            
            endImageEmbTime = time.perf_counter()

            totalImgEmbLatencyTime += (endImageEmbTime - startImageEmbTime)

            highestGPUMem = max(highestGPUMem, (torch.cuda.max_memory_allocated(device) / (1024 ** 3)))

            writer.writerow([
                dataLine["parent_asin"],
                dataLine["average_rating"],
                dataLine["rating_number"],
                dataLine["price"],
                descEmb,
                imgEmb
            ])
            
            resultCounter += 1
            print("Data line", resultCounter, "has been added to the file:", outputFileName)

            # NOTE: hard code stop after 100,000 data lines have been added to the file
            if resultCounter >= 100001:
                break

        LOGGER.info("#################################################################")
        LOGGER.info(outputFileName + " Embedding Statistics:")
        LOGGER.info("#################################################################\n")
        
        # -1 result counter to not count heading in result and convert to ms from seconds
        avgTextLatency = (totalTextEmbLatencyTime / (resultCounter - 1)) * 1000
        LOGGER.info(f"Average Text Embedding Latency Spent Per Item (ms): {avgTextLatency:.6f}")

        avgImgLatency = (totalImgEmbLatencyTime / (resultCounter - 1)) * 1000
        LOGGER.info(f"Average Image Embedding Latency Spent Per Item (ms): {avgImgLatency:.6f}")

        # get total time spent to compute all embeddings (in minutes)
        LOGGER.info(f"Total Time Spent to Perform All Text Embeddings in Dataset (minutes): {(totalTextEmbLatencyTime / 60):.6f}")
        LOGGER.info(f"Total Time Spent to Perform All Image Embeddings in Dataset (minutes): {(totalImgEmbLatencyTime / 60):.6f}")

        # get the highest GPU memory recorded during the text embedding process
        LOGGER.info(f"Highest GPU Memory Recorded During Embedding Process (GB): {highestGPUMem:.6f}\n")


if __name__ == "__main__":
    # tensors are a datatype in PyTorch used to represent information such as weights and inputs
    # try to use GPU for PyTorch tensors and models, otherwise use CPU

    # CUDA Version -> 12.9 from nvidia-smi -> using 12.8 for torch
    # torch = 2.7.1+cu128
    # torchvision = 0.22.1+cu128

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    amazonReviewFile = CURRENT_DIR + "/dataset/review_Amazon_Fashion.jsonl"
    amazonMetaFile = CURRENT_DIR + "/dataset/meta_Amazon_Fashion.jsonl"
    
    ##################################
    # dataset output directories
    ##################################

    BERT_RES50_OutputDir = CURRENT_DIR + "/atomic_data_path/BERT_RES50_Amazon_Fashion/"
    BERT_RES18_OutputDir = CURRENT_DIR + "/atomic_data_path/BERT_RES18_Amazon_Fashion/"
    BERT_MOBILENETV2_OutputDir = CURRENT_DIR + "/atomic_data_path/BERT_MOBILENETV2_Amazon_Fashion/"
    BERT_VITSMALL_OutputDir = CURRENT_DIR + "/atomic_data_path/BERT_VITSMALL_Amazon_Fashion/"


    DistilBERT_RES50_OutputDir = CURRENT_DIR + "/atomic_data_path/DistilBERT_RES50_Amazon_Fashion/"
    DistilBERT_RES18_OutputDir = CURRENT_DIR + "/atomic_data_path/DistilBERT_RES18_Amazon_Fashion/"
    DistilBERT_MOBILENETV2_OutputDir = CURRENT_DIR + "/atomic_data_path/DistilBERT_MOBILENETV2_Amazon_Fashion/"
    DistilBERT_VITSMALL_OutputDir = CURRENT_DIR + "/atomic_data_path/DistilBERT_VITSMALL_Amazon_Fashion/"


    ALBERT_RES50_OutputDir = CURRENT_DIR + "/atomic_data_path/ALBERT_RES50_Amazon_Fashion/"
    ALBERT_RES18_OutputDir = CURRENT_DIR + "/atomic_data_path/ALBERT_RES18_Amazon_Fashion/"
    ALBERT_MOBILENETV2_OutputDir = CURRENT_DIR + "/atomic_data_path/ALBERT_MOBILENETV2_Amazon_Fashion/"
    ALBERT_VITSMALL_OutputDir = CURRENT_DIR + "/atomic_data_path/ALBERT_VITSMALL_Amazon_Fashion/"


    TinyBERT_RES50_OutputDir = CURRENT_DIR + "/atomic_data_path/TinyBERT_RES50_Amazon_Fashion/"
    TinyBERT_RES18_OutputDir = CURRENT_DIR + "/atomic_data_path/TinyBERT_RES18_Amazon_Fashion/"
    TinyBERT_MOBILENETV2_OutputDir = CURRENT_DIR + "/atomic_data_path/TinyBERT_MOBILENETV2_Amazon_Fashion/"
    TinyBERT_VITSMALL_OutputDir = CURRENT_DIR + "/atomic_data_path/TinyBERT_VITSMALL_Amazon_Fashion/"


    ##################################
    # BERT .inter
    ##################################

    # setup BERT model and tokenizer
    bertModel = BertModel.from_pretrained("bert-base-uncased").to(device)

    # set model to evaluation mode and disable dropout to prevent noise being added to embeddings when our multimodal recommender uses this data
    bertModel.eval()

    # bert-base-uncased is a pretrained English model
    bertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    #buildInterFile(device, amazonReviewFile, "BERT_RES50_Amazon_Fashion.inter", BERT_RES50_OutputDir,
    #               encoders.getBERTEmb, bertModel, bertTokenizer, "text_BERT_emb")


    ##################################
    # DistilBERT .inter
    ##################################

    distilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    distilBertModel.eval()

    distilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    #buildInterFile(device, amazonReviewFile, "DistilBERT_RES50_Amazon_Fashion.inter", DistilBERT_RES50_OutputDir,
    #               encoders.getDistilBertEmb, distilBertModel, distilBertTokenizer, "text_DistilBERT_emb")


    ##################################
    # ALBERT .inter
    ##################################

    albertModel = AlbertModel.from_pretrained("albert-base-v2").to(device)
    albertModel.eval()

    albertTokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    #buildInterFile(device, amazonReviewFile, "ALBERT_RES50_Amazon_Fashion.inter", ALBERT_RES50_OutputDir,
    #               encoders.getALBERTEmb, albertModel, albertTokenizer, "text_ALBERT_emb")


    ##################################
    # TinyBERT .inter
    ##################################

    tinyBertModel = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)
    tinyBertModel.eval()

    tinyBertTokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    #buildInterFile(device, amazonReviewFile, "TinyBERT_RES50_Amazon_Fashion.inter", TinyBERT_RES50_OutputDir,
    #               encoders.getTinyBERTEmb, tinyBertModel, tinyBertTokenizer, "text_TinyBERT_emb")
    

    ##################################
    # Global .user
    ##################################

    #buildUserFile(amazonReviewFile, "BERT_RES50_Amazon_Fashion.user", BERT_RES50_OutputDir)


    ############################################################### 
    # BERT, DistilBERT, ALBERT and TinyBERT + ResNet50 .item
    ###############################################################
    
    # setup image preprocessing before feeding image into resnet50, resnet18 and mobilenetv2 model - code below produces a 3 x 224 x 224 tensor
    # image preprocessing outline and values used for mean and std are inspired from https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.quantization.resnet18.html?highlight=transforms+normalize#:~:text=weights='IMAGENET1K_FBGEMM_V1'%20.-,ResNet18_QuantizedWeights.,DEFAULT%20.&text=The%20inference%20transforms%20are%20available,0.229%2C%200.224%2C%200.225%5D%20.
    # resize -> centercrop -> rescale -> normalize using mean and std
    preprocess = transforms.Compose([

        # resize and crop to the center of the image using the same dimensions that the resnet model was trained on
        transforms.Resize(256),
        transforms.CenterCrop(224),

        # convert our PIL image to a tensor multidimensional array and rescale values to [0.0, 1.0] before normalization (as expected from docs)
        transforms.ToTensor(),

        # normalize image tensor using the same values as expected from images that resnet and mobilenetv2 trained on from ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # use resnet50 model whose weights are pretrained on the ImageNet database
    res50Model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # get the number of dimensions for the image feature embedding before the classification is done, this will usually be 2048 for resnet50
    res50EmbDim = res50Model.fc.in_features

    # remove the classification layer at the end of the model so we get the raw 2048 dimension embedding vector instead of a class score
    res50Model.fc = torch.nn.Identity()

    # move all the model components such as the weights and parameters to the device (preferably GPU, use CPU if this fails) and setup model in evaluation mode (turn off dropout)
    res50Model = res50Model.to(device)
    
    # turn off dropout since we are only doing inference (using a model to make predictions or generate output based on new, unseen data)
    res50Model.eval()

    # BERT + RESNET50
    #buildItemFile(device, amazonMetaFile, "BERT_RES50_Amazon_Fashion.item", BERT_RES50_OutputDir, encoders.getBERTEmb, bertModel, bertTokenizer, "desc_BERT_emb", 
    #              encoders.getRes50Emb, res50Model, res50EmbDim, preprocess, "img_RES50_emb", False)
    
    # DistilBERT + RESNET50
    #buildItemFile(device, amazonMetaFile, "DistilBERT_RES50_Amazon_Fashion.item", DistilBERT_RES50_OutputDir, encoders.getDistilBertEmb, distilBertModel, distilBertTokenizer, "desc_DistilBERT_emb", 
    #              encoders.getRes50Emb, res50Model, res50EmbDim, preprocess, "img_RES50_emb", False)

    # ALBERT + RESNET50
    #buildItemFile(device, amazonMetaFile, "ALBERT_RES50_Amazon_Fashion.item", ALBERT_RES50_OutputDir, encoders.getALBERTEmb, albertModel, albertTokenizer, "desc_ALBERT_emb", 
    #              encoders.getRes50Emb, res50Model, res50EmbDim, preprocess, "img_RES50_emb", False)

    # TinyBERT + RESNET50
    #buildItemFile(device, amazonMetaFile, "TinyBERT_RES50_Amazon_Fashion.item", TinyBERT_RES50_OutputDir, encoders.getTinyBERTEmb, tinyBertModel, tinyBertTokenizer, "desc_TinyBERT_emb", 
    #              encoders.getRes50Emb, res50Model, res50EmbDim, preprocess, "img_RES50_emb", False)


    ############################################################### 
    # BERT, DistilBERT, ALBERT and TinyBERT + ResNet18 .item
    ###############################################################

    res18Model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # get the number of dimensions for the image embedding - expect 512
    res18EmbDim = res18Model.fc.in_features

    res18Model.fc = torch.nn.Identity()

    res18Model = res18Model.to(device)
    res18Model.eval()

    # BERT + RESNET18
    #buildItemFile(device, amazonMetaFile, "BERT_RES18_Amazon_Fashion.item", BERT_RES18_OutputDir, encoders.getBERTEmb, bertModel, bertTokenizer, "desc_BERT_emb", 
    #              encoders.getRes18Emb, res18Model, res18EmbDim, preprocess, "img_RES18_emb", False)

    # DistilBERT + RESNET18
    #buildItemFile(device, amazonMetaFile, "DistilBERT_RES18_Amazon_Fashion.item", DistilBERT_RES18_OutputDir, encoders.getDistilBertEmb, distilBertModel, distilBertTokenizer, "desc_DistilBERT_emb", 
    #              encoders.getRes18Emb, res18Model, res18EmbDim, preprocess, "img_RES18_emb", False)

    # ALBERT + RESNET18
    #buildItemFile(device, amazonMetaFile, "ALBERT_RES18_Amazon_Fashion.item", ALBERT_RES18_OutputDir, encoders.getALBERTEmb, albertModel, albertTokenizer, "desc_ALBERT_emb", 
    #              encoders.getRes18Emb, res18Model, res18EmbDim, preprocess, "img_RES18_emb", False)

    # TinyBERT + RESNET18
    #buildItemFile(device, amazonMetaFile, "TinyBERT_RES18_Amazon_Fashion.item", TinyBERT_RES18_OutputDir, encoders.getTinyBERTEmb, tinyBertModel, tinyBertTokenizer, "desc_TinyBERT_emb", 
    #              encoders.getRes18Emb, res18Model, res18EmbDim, preprocess, "img_RES18_emb", False)


    ############################################################### 
    # BERT, DistilBERT, ALBERT and TinyBERT + MobileNetV2 .item
    ###############################################################

    mobileNetV2Model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # get the number of dimensions for the iamge embedding - expect 1280
    mobileNetV2EmbDim = mobileNetV2Model.last_channel

    # mobilenetv2 uses a sequential classifier instead of a linear fully connected layer like resnet at the final classification layer
    mobileNetV2Model.classifier = torch.nn.Identity()

    mobileNetV2Model = mobileNetV2Model.to(device)
    mobileNetV2Model.eval()

    # BERT + MOBILENETV2
    #buildItemFile(device, amazonMetaFile, "BERT_MOBILENETV2_Amazon_Fashion.item", BERT_MOBILENETV2_OutputDir, encoders.getBERTEmb, bertModel, bertTokenizer, "desc_BERT_emb", 
    #              encoders.getMobileNetV2Emb, mobileNetV2Model, mobileNetV2EmbDim, preprocess, "img_MOBILENETV2_emb", False)

    # DistilBERT + MOBILENETV2
    #buildItemFile(device, amazonMetaFile, "DistilBERT_MOBILENETV2_Amazon_Fashion.item", DistilBERT_MOBILENETV2_OutputDir, encoders.getDistilBertEmb, distilBertModel, distilBertTokenizer, "desc_DistilBERT_emb", 
    #              encoders.getMobileNetV2Emb, mobileNetV2Model, mobileNetV2EmbDim, preprocess, "img_MOBILENETV2_emb", False)

    # ALBERT + MOBILENETV2
    #buildItemFile(device, amazonMetaFile, "ALBERT_MOBILENETV2_Amazon_Fashion.item", ALBERT_MOBILENETV2_OutputDir, encoders.getALBERTEmb, albertModel, albertTokenizer, "desc_ALBERT_emb", 
    #              encoders.getMobileNetV2Emb, mobileNetV2Model, mobileNetV2EmbDim, preprocess, "img_MOBILENETV2_emb", False)

    # TinyBERT + MOBILENETV2
    #buildItemFile(device, amazonMetaFile, "TinyBERT_MOBILENETV2_Amazon_Fashion.item", TinyBERT_MOBILENETV2_OutputDir, encoders.getTinyBERTEmb, tinyBertModel, tinyBertTokenizer, "desc_TinyBERT_emb", 
    #              encoders.getMobileNetV2Emb, mobileNetV2Model, mobileNetV2EmbDim, preprocess, "img_MOBILENETV2_emb", False)

    
    ############################################################### 
    # BERT, DistilBERT, ALBERT and TinyBERT + ViT-Small (16) .item
    ###############################################################

    vitSmallModel = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224")

    # remove classification layer at the end of the model so we get the raw 384 dimension embedding
    vitSmallModel.classifier = torch.nn.Identity()

    # move model to GPU if available and turn off dropout since we are just using the model to make predictions on unseen images
    vitSmallModel = vitSmallModel.to(device)
    vitSmallModel.eval()

    # setup vit-small image processor to use optional fast image processor class instead of slow image processor class
    vitPreProcess = AutoImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224", use_fast=True)

    # BERT + VITSMALL
    #buildItemFile(device, amazonMetaFile, "BERT_VITSMALL_Amazon_Fashion.item", BERT_VITSMALL_OutputDir, encoders.getBERTEmb, bertModel, bertTokenizer, "desc_BERT_emb", 
    #              encoders.getVitSmallEmb, vitSmallModel, None, vitPreProcess, "img_VITSMALL_emb", True)

    # DistilBERT + VITSMALL
    #buildItemFile(device, amazonMetaFile, "DistilBERT_VITSMALL_Amazon_Fashion.item", DistilBERT_VITSMALL_OutputDir, encoders.getDistilBertEmb, distilBertModel, distilBertTokenizer, "desc_DistilBERT_emb", 
    #              encoders.getVitSmallEmb, vitSmallModel, None, vitPreProcess, "img_VITSMALL_emb", True)

    # ALBERT + VITSMALL
    #buildItemFile(device, amazonMetaFile, "ALBERT_VITSMALL_Amazon_Fashion.item", ALBERT_VITSMALL_OutputDir, encoders.getALBERTEmb, albertModel, albertTokenizer, "desc_ALBERT_emb", 
    #              encoders.getVitSmallEmb, vitSmallModel, None, vitPreProcess, "img_VITSMALL_emb", True)

    # TinyBERT + VITSMALL
    #buildItemFile(device, amazonMetaFile, "TinyBERT_VITSMALL_Amazon_Fashion.item", TinyBERT_VITSMALL_OutputDir, encoders.getTinyBERTEmb, tinyBertModel, tinyBertTokenizer, "desc_TinyBERT_emb", 
    #              encoders.getVitSmallEmb, vitSmallModel, None, vitPreProcess, "img_VITSMALL_emb", True)


    ############################################################### 
    # Old Function Test Cases - No longer functional
    ###############################################################

    #buildInterFile(amazonReviewFile, "Amazon_Fashion.inter")

    #buildUserFile(amazonReviewFile, "Amazon_Fashion.user")

    #buildItemFile(amazonMetaFile, "Amazon_Fashion.item")