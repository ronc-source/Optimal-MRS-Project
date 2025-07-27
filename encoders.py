# libraries
import json
import csv
import torch
import requests
import io

from PIL import Image
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, AlbertModel, AlbertTokenizer, AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights


# constants

# max amount of text tokens we will feed to our BERT text encoders
MAX_TEXT_TOKEN_LENGTH = 128

########################
# text encoders
########################

# BERT
def getBERTEmb(text, device, maxLength=MAX_TEXT_TOKEN_LENGTH):
    # 1. setup BERT

    # bert model is moved to device, so its weights and any new tensors now live on GPU if available or CPU otherwise
    bertModel = BertModel.from_pretrained("bert-base-uncased").to(device)

    # set model to evaluation mode and disable dropout to prevent noise being added to embeddings when our multimodal recommender uses this data
    bertModel.eval()

    # bert-base-uncased is a pretrained English model
    bertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 2. get text embeddings

    text = text.strip()

    # text is empty, return zero vector of same expected dimension
    if not text:

        # model.config.hidden_size is the 768 dimension embedding
        zeroTensor = torch.zeros(bertModel.config.hidden_size, device=device)

        zeroRes = []
        for i in zeroTensor:

            # convert zero to 6 decimal places
            zeroRes.append(f"{i:.6f}")
        
        # separate embedding values with white space to conform to recbole atomic file expectations of float sequences
        return " ".join(zeroRes)
    
    # prepare a tensor of input tokens for BERT model
    formatInput = bertTokenizer(
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

    # disable gradient tracking in model as it is not required since we are directly generating text embeddings for our recommender
    with torch.no_grad():

        # return a tensor of shape (batchSize, seqLen, hiddenSize)
        res = bertModel(**inputDict)
    
    # get the CLS 768 dimension text embedding and put it on our device
    textEmb = res.last_hidden_state[0, 0, :].to(device)

    print("Length of BERT embedding", len(textEmb))

    textEmbRes = []
    for i in textEmb:

        # format text embedding values to 6 decimal places
        textEmbRes.append(f"{i:.6f}")
    
    # return 768 dim text embedding as a string of values separated by white space to conform to recbole atomic file expectations of float sequence
    return " ".join(textEmbRes)


# DistilBERT
def getDistilBertEmb(text, device, maxLength=MAX_TEXT_TOKEN_LENGTH):
    # 1. setup DistilBERT
    
    distilBertModel = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    distilBertModel.eval()

    distilBertTokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # 2. get text embeddings

    text = text.strip()

    if not text:
        # model.config.hidden_size should also be 768 dimension embedding
        zeroTensor = torch.zeros(distilBertModel.config.hidden_size, device=device)

        zeroRes = []
        for i in zeroTensor:
            zeroRes.append(f"{i:.6f}")
        
        # separate embedding values with white space to conform to recbole atomic file expectations
        return " ".join(zeroRes)
    
    formatInput = distilBertTokenizer(
        text,
        max_length = maxLength,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )

    inputDict = {}

    for key, val in formatInput.items():
        inputDict[key] = val.to(device)

    with torch.no_grad():
        res = distilBertModel(**inputDict)

    textEmb = res.last_hidden_state[0, 0, :].to(device)

    print("Length of DistilBERT embedding", len(textEmb))

    textEmbRes = []
    for i in textEmb:
        textEmbRes.append(f"{i:.6f}")
    
    # return 768 dim text embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(textEmbRes)


# ALBERT
def getALBERTEmb(text, device, maxLength=MAX_TEXT_TOKEN_LENGTH):
    # 1. setup ALBERT

    albertModel = AlbertModel.from_pretrained("albert-base-v2").to(device)
    albertModel.eval()

    albertTokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

    # 2. get text embeddings
    
    text = text.strip()

    if not text:
        # model.config.hidden_size should also be 768 dimension embedding
        zeroTensor = torch.zeros(albertModel.config.hidden_size, device=device)

        zeroRes = []
        for i in zeroTensor:
            zeroRes.append(f"{i:.6f}")

        # separate embedding values with white space to conform to recbole atomic file expectations
        return " ".join(zeroRes)

    formatInput = albertTokenizer(
        text,
        max_length = maxLength,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )

    inputDict = {}

    for key, val in formatInput.items():
        inputDict[key] = val.to(device)
    
    with torch.no_grad():
        res = albertModel(**inputDict)
    
    textEmb = res.last_hidden_state[0, 0, :].to(device)

    print("Length of ALBERT embedding", len(textEmb))

    textEmbRes = []
    for i in textEmb:
        textEmbRes.append(f"{i:.6f}")
    
    # return 768 dim text embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(textEmbRes)


# TinyBERT
def getTinyBERTEmb(text, device, maxLength=MAX_TEXT_TOKEN_LENGTH):
    # 1. setup TinyBERT

    tinyBertModel = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D").to(device)
    tinyBertModel.eval()

    tinyBertTokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

    # 2. get text embeddings

    text = text.strip()

    if not text:
        #model.config.hidden_size should be 312 dimension embedding
        zeroTensor = torch.zeros(tinyBertModel.config.hidden_size, device=device)

        zeroRes = []
        for i in zeroTensor:
            zeroRes.append(f"{i:.6f}")

        # separate embedding values with white space to conform to recbole atomic file expectations
        return " ".join(zeroRes)

    formatInput = tinyBertTokenizer(
        text,
        max_length = maxLength,
        padding = "max_length",
        truncation = True,
        return_tensors = "pt"
    )

    inputDict = {}

    for key, val in formatInput.items():
        inputDict[key] = val.to(device)
    
    with torch.no_grad():
        res = tinyBertModel(**inputDict)
    
    textEmb = res.last_hidden_state[0, 0, :].to(device)

    print("Length of TinyBERT embedding", len(textEmb))
    
    textEmbRes = []
    for i in textEmb:
        textEmbRes.append(f"{i:.6f}")
    
    # return 312 dim text embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(textEmbRes)


########################
# image encoders
########################

# ResNet-50
def getRes50Emb(imgURL, device):
    # 1. setup resnet50

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

    # 2. setup image preprocessing requirements for resnet50

    # setup image preprocessing before feeding image into resnet 50 model - code below produces a 3 x 224 x 224 tensor
    # image preprocessing outline and values used for mean and std are inspired from https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.quantization.resnet18.html?highlight=transforms+normalize#:~:text=weights='IMAGENET1K_FBGEMM_V1'%20.-,ResNet18_QuantizedWeights.,DEFAULT%20.&text=The%20inference%20transforms%20are%20available,0.229%2C%200.224%2C%200.225%5D%20.
    # resize -> centercrop -> rescale -> normalize using mean and std
    preprocess = transforms.Compose([

        # resize and crop to the center of the image using the same dimensions that the resnet model was trained on
        transforms.Resize(256),
        transforms.CenterCrop(224),

        # convert our PIL image to a tensor multidimensional array and rescale values to [0.0, 1.0] before normalization (as expected from docs)
        transforms.ToTensor(),

        # normalize image tensor using the same values as expected from images that resnet trained on from ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # 3. try to download the image from the URL, preprocess it and feed it into the resnet50 model - if there are any issues return a zero vector of size 2048

    try:
        # download with a 10 second timeout to avoid broken URLs
        imgReq = requests.get(imgURL, timeout=10)

        # convert PIL image from bytes to an RGB image as resnet50 does not accept grayscale and requires 3 channels (RGB)
        imgFound = Image.open(io.BytesIO(imgReq.content)).convert("RGB")

        # preprocess image found from dataset
        # add a batch dimension to get tensor shape as [1, 3, 224, 224] from [3, 224, 224] for resnet50 model input requirements
        imgInput = preprocess(imgFound).unsqueeze(0)

        # move result tensor to GPU if available, CPU otherwise
        imgInput = imgInput.to(device)

        # we do not need to compute or track gradients as we are just calling the model to produce embeddings
        with torch.no_grad():
            # get a 2048 dimension vector embedding from resnet50
            imgEmb = res50Model(imgInput)

        # remove front batch dimension to go from [1, 2048] to [2048] shape using squeeze and convert tensor to a list of float values
        imgEmb = imgEmb.squeeze(0).tolist()

    except Exception:
        # in case of any issues with the image embedding, use a zero vector
        imgEmb = [0.0] * res50EmbDim
    
    print("Length of ResNet50 embedding", len(imgEmb))
    
    imgEmbRes = []

    for i in imgEmb:
        # set the float value to 6 decimal places
        imgEmbRes.append(f"{i:.6f}")
    
    # return a 2048 dim image embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(imgEmbRes)


# ResNet-18
def getRes18Emb(imgURL, device):
    # 1. setup resnet18
    
    res18Model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # get the number of dimensions for the image embedding - expect 512
    res18EmbDim = res18Model.fc.in_features

    res18Model.fc = torch.nn.Identity()

    res18Model = res18Model.to(device)
    res18Model.eval()

    # 2. setup image preprocessing requirements for resnet18 - same as resnet50

    preprocess = transforms.Compose([

        # resize and crop to the center of the image using the same dimensions that the resnet model was trained on
        transforms.Resize(256),
        transforms.CenterCrop(224),

        # convert our PIL image to a tensor multidimensional array and rescale values to [0.0, 1.0] before normalization (as expected from docs)
        transforms.ToTensor(),

        # normalize image tensor using the same values as expected from images that resnet trained on from ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # 3. try to download the image from the URL, preprocess it and feed it into the resnet18 model - if there are any issues return a zero vector
    try:
        imgReq = requests.get(imgURL, timeout=10)

        imgFound = Image.open(io.BytesIO(imgReq.content)).convert("RGB")

        # unsqueeze to get tensor shape as [1, 3, 224, 224] for resnet18 model input requirements
        # 1 represents the first and only batch we will use to represent the 3-channel (RGB) 224x224 image
        imgInput = preprocess(imgFound).unsqueeze(0)
        imgInput = imgInput.to(device)

        with torch.no_grad():
            # get a 512 dimension vector embedding from resnet18
            imgEmb = res18Model(imgInput)
        
        # go from [1, 512] to [512] shape
        imgEmb = imgEmb.squeeze(0).tolist()
    
    except Exception:
        # in case of any issues with the image embedding process, use a zero vector with dimension 512
        imgEmb = [0.0] * res18EmbDim
    
    print("Length of ResNet18 embedding", len(imgEmb))

    imgEmbRes = []

    for i in imgEmb:
        imgEmbRes.append(f"{i:.6f}")
    
    # return a 512 dim image embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(imgEmbRes)


# MobileNetV2
def getMobileNetV2Emb(imgURL, device):
    # 1. setup mobilenetv2

    mobileNetV2Model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # get the number of dimensions for the iamge embedding - expect 1280
    mobileNetV2EmbDim = mobileNetV2Model.last_channel

    # mobilenetv2 uses a sequential classifier instead of a linear fully connected layer like resnet at the final classification layer
    mobileNetV2Model.classifier = torch.nn.Identity()

    mobileNetV2Model = mobileNetV2Model.to(device)
    mobileNetV2Model.eval()

    # 2. setup image preprocessing requirements for mobilenetv2 - same as resnet (all trained on imagenet)

    preprocess = transforms.Compose([

        # resize and crop to the center of the image using the same dimensions that the mobilenetv2 model was trained on
        transforms.Resize(256),
        transforms.CenterCrop(224),

        # convert our PIL image to a tensor multidimensional array and rescale values to [0.0, 1.0] before normalization (as expected from docs)
        transforms.ToTensor(),

        # normalize image tensor using the same values as expected from images that mobilenetv2 trained on from ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # 3. try to download the image from the URL, preprocess it and feed it into the mobilenetv2 model - if there are any issues return a zero vector
    try:
        imgReq = requests.get(imgURL, timeout=10)

        imgFound = Image.open(io.BytesIO(imgReq.content)).convert("RGB")

        # unsqueeze to get tensor shape as [1, 3, 224, 224] for mobilenetv2 model input requirements
        # 1 represents the first and only batch we will use to represent the 3-channel (RGB) 224x224 image
        imgInput = preprocess(imgFound).unsqueeze(0)
        imgInput = imgInput.to(device)

        with torch.no_grad():
            # get a 1280 dimension vector embedding from mobilenetv2
            imgEmb = mobileNetV2Model(imgInput)
        
        # go from [1, 1280] to [1280] shape
        imgEmb = imgEmb.squeeze(0).tolist()
    
    except Exception:
        # in case of any issues with the image embedding process, use a zero vector with dimension 512
        imgEmb = [0.0] * mobileNetV2EmbDim
    
    print("Length of MobileNetV2 embedding", len(imgEmb))

    imgEmbRes = []

    for i in imgEmb:
        imgEmbRes.append(f"{i:.6f}")
    
    # return a 1280 dim image embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(imgEmbRes)


# Vit-Small (16 x 16) patches
def getVitSmallEmb(imgURL, device):
    # 1. setup vit-small

    vitSmallModel = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-small-patch16-224")

    # remove classification layer at the end of the model so we get the raw 384 dimension embedding
    vitSmallModel.classifier = torch.nn.Identity()

    # move model to GPU if available and turn off dropout since we are just using the model to make predictions on unseen images
    vitSmallModel = vitSmallModel.to(device)
    vitSmallModel.eval()

    # setup vit-small image processor to use optional fast image processor class instead of slow image processor class
    preprocess = AutoImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224", use_fast=True)
    
    try:
        imgReq = requests.get(imgURL, timeout=10)

        imgFound = Image.open(io.BytesIO(imgReq.content)).convert("RGB")

        processImage = preprocess(
            imgFound,
            return_tensors = "pt"
        )

        inputDict = {}

        for key, val in processImage.items():
            inputDict[key] = val.to(device)
        
        # anything in this block will disable gradient tracking, which we do not need as we are just generating the image embedding
        with torch.no_grad():
            res = vitSmallModel(**inputDict)
        
        # access vit-small output tensor list of embedding values in first index of logits and convert to an array for further processing
        imgEmb = res.logits[0].tolist()
    
    except Exception:
        # model.config.hidden_size should be 384 dimension embedding
        imgEmb = torch.zeros(vitSmallModel.config.hidden_size, device=device)
    
    print("Length of Vit-Small embedding", len(imgEmb))

    imgEmbRes = []
    for i in imgEmb:
        imgEmbRes.append(f"{i:.6f}")
    
    # return 384 dim image embedding with white space separation between vector values to conform to recbole atomic file expectations
    return " ".join(imgEmbRes)

    

# main
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    

    # Test text encoders 

    #res = getBERTEmb("this is a test", device)
    #print(res)

    #res = getDistilBertEmb("this is a test", device)
    #print(res)

    #res = getALBERTEmb("this is a test", device)
    #print(res)

    #res = getTinyBERTEmb("this is a test", device)
    #print(res)


    # Test image encoders

    # from sample data row of Meta Amazon Fashion dataset - https://amazon-reviews-2023.github.io/
    testImgURL = "https://m.media-amazon.com/images/I/31dlCd7tHSL.jpg"

    #res = getRes50Emb(testImgURL, device)
    #print(res)

    #res = getRes18Emb(testImgURL, device)
    #print(res)

    #res = getMobileNetV2Emb(testImgURL, device)
    #print(res)

    res = getVitSmallEmb(testImgURL, device)
    print(res)


    # Test device

    # prints "cuda"
    #print(device)