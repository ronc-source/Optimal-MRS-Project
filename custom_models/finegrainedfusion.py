import torch
from torch import nn
#from recbole.model.context_aware_recommender import FM

# imports from recbole doc for new model creation
from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import ContextRecommender
import torch.nn.functional

import math

# from fm.py
from torch.nn.init import xavier_normal_
from recbole.model.layers import BaseFactorizationMachine
from recbole.model.layers import FMFirstOrderLinear


class FineGrainedFusion(ContextRecommender):

    input_type = InputType.PAIRWISE
    

    def __init__(self, config, dataset):
        # explicitly declare neg_parent_asin and associate it with existing itemID's
        self.negItemIDField = config["NEG_ITEM_ID_FIELD"]
        dataset.field2id_token[self.negItemIDField] = dataset.field2id_token[config["ITEM_ID_FIELD"]]

        super(FineGrainedFusion, self).__init__(config, dataset)

        self.dataset = dataset

        self.numUsers = dataset.user_num
        self.numItems = dataset.item_num

        # get the embedding size we defined in our .yaml config - typically 64 based on FM models
        self.embSize = config["embedding_size"]

        # get user and parent ID field
        self.userIDField = config["USER_ID_FIELD"]
        self.itemIDField = config["ITEM_ID_FIELD"]

        # get the main text and image modality encoder feature fields from .inter and .item
        self.textField = config["text_field"]
        self.descField = config["desc_field"]
        self.imgField = config["img_field"]

        # get the side-feature fields - do not include rating as it is used as a threshold column for the model
        self.timestampField = "timestamp"
        self.averageRatingField = "average_rating"
        self.ratingNumberField = "rating_number"
        self.priceField = "price"

        # get the dimensions of the text and image modality encoder feature fields from .inter and .item
        textDim = config[self.textField] # 768
        descDim = config[self.descField] # 768
        imgDim = config[self.imgField] # 2048

        # setup projection layer for text and image modality embeddings into our declared embedding size (64)
        self.textProjectionLayer = nn.Linear(textDim, self.embSize)
        self.descProjectionLayer = nn.Linear(descDim, self.embSize)
        self.imgProjectionLayer = nn.Linear(imgDim, self.embSize)

        # setup projection layer for the side-feature fields
        self.timestampProjectionLayer = nn.Linear(1, self.embSize)
        self.averageRatingProjectionLayer = nn.Linear(1, self.embSize)
        self.ratingNumberProjectionLayer = nn.Linear(1, self.embSize)
        self.priceProjectionLayer = nn.Linear(1, self.embSize)

        # define amount of chunks/sub-vectors to separate embedding after projection (must be evenly divisible by 64 - our embSize)
        self.chunkAmount = 8

        # set the chunk/sub-vector dimension, which should be the embSize (64) / the amount of chunks (8) = 8
        self.chunkDim = self.embSize // self.chunkAmount

        '''
        # setup auto encoder layers for img denoising
        self.encodeFirstLayer = nn.Linear(imgDim, imgDim // 2) # go from 2048 -> 1024
        self.encodeSecondLayer = nn.Linear(imgDim // 2, imgDim // 4) # go from 1024 -> 512
        self.encodeThirdLayer = nn.Linear(imgDim // 4, imgDim // 8) # go from 512 -> 256
        self.encodeFourthLayer = nn.Linear(imgDim // 8, imgDim // 16) # go from 256 -> 128

        self.decodeFirstLayer = nn.Linear(imgDim // 16, imgDim // 8) # go from 128 -> 256
        self.decodeSecondLayer = nn.Linear(imgDim // 8, imgDim // 4) # go from 256 -> 512
        self.decodeThirdLayer = nn.Linear(imgDim // 4, imgDim // 2) # go from 512 -> 1024
        self.decodeFourthLayer = nn.Linear(imgDim // 2, imgDim) # go from 1024 -> 2048

        self.relu = nn.ReLU()
        '''

        # define the layers and loss function
        self.userEmbedding = nn.Embedding(self.numUsers, self.embSize)
        self.itemEmbedding = nn.Embedding(self.numItems, self.embSize)
        self.loss = BPRLoss()
        self.sigmoid = nn.Sigmoid()
        self.finalFusionLayer = nn.Linear(self.embSize, 1)

        # intialize parameters in accordance with guidelines for new models in recbole
        self.apply(xavier_normal_initialization)

    
    '''
    # setup encoder for img denoising
    def encode(self, inputVector):
        resOne = self.relu(self.encodeFirstLayer(inputVector))
        resTwo = self.relu(self.encodeSecondLayer(resOne))
        resThree = self.relu(self.encodeThirdLayer(resTwo))
        resFour = self.relu(self.encodeFourthLayer(resThree))

        return resFour
    

    # setup decoder for img denoising
    def decode(self, inputVector):

        resOne = self.relu(self.decodeFirstLayer(inputVector))
        resTwo = self.relu(self.decodeSecondLayer(resOne))
        resThree = self.relu(self.decodeThirdLayer(resTwo))
        resFour = self.relu(self.decodeFourthLayer(resThree))

        return self.sigmoid(resFour)
    '''
    
    def forward(self, interaction):
        # get the user and item ID embeddings
        userID = interaction[self.userIDField] # torch.Size([303])
        itemID = interaction[self.itemIDField] # torch.Size([303])

        userIDEmb = self.userEmbedding(userID) # torch.Size([303, 64])
        itemIDEmb = self.itemEmbedding(itemID) # torch.Size([303, 64])

        # get the text and image modality embeddings from the atomic files
        textData = interaction[self.textField] # torch.Size([303, 768])
        descData = interaction[self.descField] # torch.Size([303, 768])
        imgData = interaction[self.imgField] # torch.Size([303, 2048])

        # apply auto encoder for denoising image embeddings
        # encodedImg = self.encode(imgData)
        # imgData = self.decode(encodedImg)


        # get the side-features from the atomic files
        timestampData = interaction[self.timestampField] # torch.Size([303])
        timestampData = timestampData.unsqueeze(-1) # torch.Size([303, 1])

        averageRatingData = interaction[self.averageRatingField] # torch.Size([303])
        averageRatingData = averageRatingData.unsqueeze(-1) # torch.Size([303, 1])

        ratingNumberData = interaction[self.ratingNumberField] # torch.Size([303])
        ratingNumberData = ratingNumberData.unsqueeze(-1) # torch.Size([303, 1])

        priceData = interaction[self.priceField] # torch.Size([303])
        priceData = priceData.unsqueeze(-1) # torch.Size([303, 1])

        #print("CHECK 1 TEST")
        # apply projection layer to text and image modality embeddings
        textEmb = self.textProjectionLayer(textData) # [303, 64]
        #print("TEXT EMB SHAPE:", textEmb.shape)
        descEmb = self.descProjectionLayer(descData) # [303, 64]
        #print("DESC EMB SHAPE:", descEmb.shape)
        imgEmb = self.imgProjectionLayer(imgData) # [303, 64]
        #print("IMG EMB SHAPE:", imgEmb.shape)

        #print("CHECK 2 TEST")
        # apply projection layer to side-features
        timestampEmb = self.timestampProjectionLayer(timestampData) # [303, 64]
        #print("TIMESTAMP EMB SHAPE:", timestampEmb.shape)
        averageRatingEmb = self.averageRatingProjectionLayer(averageRatingData) # [303, 64]
        #print("AVERAGE RATING EMB SHAPE:", averageRatingEmb.shape)
        ratingNumberEmb = self.ratingNumberProjectionLayer(ratingNumberData) # [303, 64]
        #print("RATING NUMBER EMB SHAPE", ratingNumberEmb.shape)
        priceEmb = self.priceProjectionLayer(priceData) # [303, 64]
        #print("PRICE EMB SHAPE", priceEmb.shape)

        # split text and image modality embedding vectors into sub-vectors [303, 64] -> [303, 8, 8]
        batchTotal = textEmb.size(0) # 303
        #print("TEST BATCH TOTAL:", batchTotal)

        textSubVec = textEmb.view(batchTotal, self.chunkAmount, self.chunkDim) # [303, 8, 8]
        #print("TEST TEXT EMB SUB VEC:", textSubVec)
        #print("TEST TEXT EMB SUB VEC SHAPE:", textSubVec.shape)

        descSubVec = descEmb.view(batchTotal, self.chunkAmount, self.chunkDim) # [303, 8, 8]
        #print("TEST DESC EMB SUB VEC:", descSubVec)
        #print("TEST DESC EMB SUB VEC SHAPE:", descSubVec.shape)

        imgSubVec = imgEmb.view(batchTotal, self.chunkAmount, self.chunkDim) # [303, 8, 8]
        #print("TEST IMG EMB SUB VEC", imgSubVec)
        #print("TEST IMG EMB SUB VEC SHAPE", imgSubVec.shape)

        # compute compatability scores between every text and image sub vector using the scaled dot product

        ''' Original implementation
        # imgSubVec.transpose [303, 8, 8] -> [303, 8, 8] (swap last 2 places)
        textAndImgCompatabilityScores = torch.matmul(textSubVec, imgSubVec.transpose(1, 2)) / math.sqrt(self.chunkDim) # [303, 8 (text), 8 (img)]
        # normalize over the image subvectors for each text sub vector (dim = 2 -> image sub vector)
        textAndImgAttentionWeights = torch.nn.functional.softmax(textAndImgCompatabilityScores, dim=2) # [303, 8 (text), 8(img)]
        # in each text sub vector get the highest signal/representation image sections from that sub vector 
        textAndImgAttend = torch.matmul(textAndImgAttentionWeights, imgSubVec)
        # fuse back the text sub vectors and prioritized/high signal image features
        textAndImageFusedSubVectors = textSubVec + textAndImgAttend # [303, 8 (chunk amount), 8 (chunk dim)]
        '''

        textAndImageFusedSubVectors = torch.nn.functional.scaled_dot_product_attention(
            query=textSubVec,
            key=imgSubVec,
            value=imgSubVec
        )

        #print("TEST TORCH TEXT AND IMG FUSED SUB VEC", textAndImageFusedSubVectors)
        #print("TEST TORCH TEXT AND IMG FUSED SUB VEC SHAPE", textAndImageFusedSubVectors.shape) # [303, 8, 8]

        descAndImageFusedSubVectors = torch.nn.functional.scaled_dot_product_attention(
            query=descSubVec,
            key=imgSubVec,
            value=imgSubVec
        )

        #print("TEST TORCH DESC AND IMG FUSED SUB VEC", descAndImageFusedSubVectors)
        #print("TEST TORCH DESC AND IMG FUSED SUB VEC SHAPE", descAndImageFusedSubVectors.shape) # [303, 8, 8]

        # bring back to expected score shape [303, 64], where 64 = declared emb size from config,  from [303, 8 (chunk amount), 8 (chunk dim)]
        textAndImageFinalFuse = textAndImageFusedSubVectors.view(batchTotal, self.chunkAmount * self.chunkDim) # [303, 64]
        descAndImageFinalFuse = descAndImageFusedSubVectors.view(batchTotal, self.chunkAmount * self.chunkDim) # [303, 64]

        finalFusion = userIDEmb + itemIDEmb + textEmb + descEmb + textAndImageFinalFuse + descAndImageFinalFuse + timestampEmb + averageRatingEmb + ratingNumberEmb + priceEmb # [303, 64]

        #print("FINAL FUSION VECTOR TEST", finalFusion)
        #print("FINAL FUSION VECTOR SHAPE TEST", finalFusion.shape) # [303, 64]'

        res = self.finalFusionLayer(finalFusion) # [303, 1]
        return res.squeeze(-1) # [303]




    # from the docs - used to compute the loss, where the input parameters are Interaction
    # returns a torch.Tensor for computing the BP information
    def calculate_loss(self, interaction):
        user = interaction[self.userIDField]
        pos_item = interaction[self.itemIDField]
        #print("DID YOU BREAK HERE?")
        neg_item = interaction[self.negItemIDField]
        #print("DID YOU BREAK NOW?")

        user_e = self.userEmbedding(user)                        # [batch_size, embedding_size]
        pos_item_e = self.itemEmbedding(pos_item)                # [batch_size, embedding_size]
        neg_item_e = self.itemEmbedding(neg_item)                # [batch_size, embedding_size]
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1) # [batch_size]
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1) # [batch_size]

        loss = self.loss(pos_item_score, neg_item_score)          # []

        return loss
    


    # from the docs - used to compute the score for a given user-item pair

    def predict(self, interaction):
        #print("INTERACTION TEXT")
        forwardedInteraction = self.forward(interaction)
        #print("TEXT INTERACTION RESULT FROM FORWARD CALL", forwardedInteraction)
        #print("TEST INTERACTION FORWARDED SHAPE", forwardedInteraction.shape)

        #print("PREDICT TEST")
        #res = self.sigmoid(self.forward(interaction))
        #print("TEST PREDICT RESULT FROM FORWARD INTERACTION", res)
        #print("TEST PREDICT SCORE SHAPE", res.shape)
        return forwardedInteraction