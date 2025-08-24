import torch
from torch import nn

from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import ContextRecommender
import torch.nn.functional

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

        # define the layers and loss function
        self.userEmbedding = nn.Embedding(self.numUsers, self.embSize)
        self.itemEmbedding = nn.Embedding(self.numItems, self.embSize)
        self.loss = BPRLoss()
        self.finalFusionLayer = nn.Linear(self.embSize, 1)

        # intialize parameters in accordance with guidelines for new models in recbole
        self.apply(xavier_normal_initialization)


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

        # get the side-features from the atomic files
        timestampData = interaction[self.timestampField] # torch.Size([303])
        timestampData = timestampData.unsqueeze(-1) # torch.Size([303, 1])

        averageRatingData = interaction[self.averageRatingField] # torch.Size([303])
        averageRatingData = averageRatingData.unsqueeze(-1) # torch.Size([303, 1])

        ratingNumberData = interaction[self.ratingNumberField] # torch.Size([303])
        ratingNumberData = ratingNumberData.unsqueeze(-1) # torch.Size([303, 1])

        priceData = interaction[self.priceField] # torch.Size([303])
        priceData = priceData.unsqueeze(-1) # torch.Size([303, 1])

        # apply projection layer to text and image modality embeddings
        textEmb = self.textProjectionLayer(textData) # [303, 64]
        descEmb = self.descProjectionLayer(descData) # [303, 64]
        imgEmb = self.imgProjectionLayer(imgData) # [303, 64]

        # apply projection layer to side-features
        timestampEmb = self.timestampProjectionLayer(timestampData) # [303, 64]
        averageRatingEmb = self.averageRatingProjectionLayer(averageRatingData) # [303, 64]
        ratingNumberEmb = self.ratingNumberProjectionLayer(ratingNumberData) # [303, 64]
        priceEmb = self.priceProjectionLayer(priceData) # [303, 64]

        # split text and image modality embedding vectors into sub-vectors [303, 64] -> [303, 8, 8]
        batchTotal = textEmb.size(0) # 303

        textSubVec = textEmb.view(batchTotal, self.chunkAmount, self.chunkDim) # [303, 8, 8]
        descSubVec = descEmb.view(batchTotal, self.chunkAmount, self.chunkDim) # [303, 8, 8]
        imgSubVec = imgEmb.view(batchTotal, self.chunkAmount, self.chunkDim) # [303, 8, 8]

        # compute compatability scores between every text and image sub vector using the scaled dot product
        textAndImageFusedSubVectors = torch.nn.functional.scaled_dot_product_attention(
            query=textSubVec,
            key=imgSubVec,
            value=imgSubVec
        )

        descAndImageFusedSubVectors = torch.nn.functional.scaled_dot_product_attention(
            query=descSubVec,
            key=imgSubVec,
            value=imgSubVec
        )

        # bring back to expected score shape [303, 64], where 64 = declared emb size from config,  from [303, 8 (chunk amount), 8 (chunk dim)]
        textAndImageFinalFuse = textAndImageFusedSubVectors.view(batchTotal, self.chunkAmount * self.chunkDim) # [303, 64]
        descAndImageFinalFuse = descAndImageFusedSubVectors.view(batchTotal, self.chunkAmount * self.chunkDim) # [303, 64]

        finalFusion = userIDEmb + itemIDEmb + textEmb + descEmb + textAndImageFinalFuse + descAndImageFinalFuse + timestampEmb + averageRatingEmb + ratingNumberEmb + priceEmb # [303, 64]

        res = self.finalFusionLayer(finalFusion) # [303, 1]
        return res.squeeze(-1) # [303]


    # inspired from the RecBole docs, used to compute the loss, where the input parameter is Interaction - https://recbole.io/docs/developer_guide/customize_models.html
    # returns a torch.Tensor for computing the BP information
    def calculate_loss(self, interaction):
        user = interaction[self.userIDField]
        pos_item = interaction[self.itemIDField]
        neg_item = interaction[self.negItemIDField]

        user_e = self.userEmbedding(user)                           # [batch_size, embedding_size]
        pos_item_e = self.itemEmbedding(pos_item)                   # [batch_size, embedding_size]
        neg_item_e = self.itemEmbedding(neg_item)                   # [batch_size, embedding_size]
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)   # [batch_size]
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)   # [batch_size]

        loss = self.loss(pos_item_score, neg_item_score)            # []

        return loss
    

    # inspired from the RecBole implementation of fm.py
    # input is an interaction and the output is a score
    def predict(self, interaction):
        forwardedInteraction = self.forward(interaction)
        return forwardedInteraction