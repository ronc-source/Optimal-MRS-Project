import torch
from torch import nn
#from recbole.model.context_aware_recommender import FM

# imports from recbole doc for new model creation
from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import ContextRecommender

# from fm.py
from torch.nn.init import xavier_normal_
from recbole.model.layers import BaseFactorizationMachine
from recbole.model.layers import FMFirstOrderLinear


# NOTE:
# gated sum fusion is only performed on text and image modality encoder embeddings
# all other features such as average_rating can be projected and concatenated without a gate
# **projection is important to convert features into the embedding space supported by the FM model

class GatedSumFusion(ContextRecommender):

    input_type = InputType.PAIRWISE
    

    def __init__(self, config, dataset):
        # explicitly declare neg_parent_asin and associate with existing itemID's
        self.negItemIDField = config["NEG_ITEM_ID_FIELD"]
        dataset.field2id_token[self.negItemIDField] = dataset.field2id_token[config["ITEM_ID_FIELD"]]

        super(GatedSumFusion, self).__init__(config, dataset)

        # reference dataset used
        self.dataset = dataset

        # get the dataset info such as the number of users and items
        self.numUsers = dataset.user_num
        self.numItems = dataset.item_num

         # get the embedding size we defined in our .yaml config for the FM model
        self.embSize = config["embedding_size"]

        # create reference to user and item ID fields using config file
        self.userIDField = config["USER_ID_FIELD"]
        self.itemIDField = config["ITEM_ID_FIELD"]

        # get the main feature fields
        self.textField = "text_BERT_emb"
        self.descField = "desc_BERT_emb"
        self.imgField = "img_RES50_emb"

        # get the side-feature fields
        # rating, timestamp, average_rating, rating_number, price

        #self.ratingField = "rating"
        self.timestampField = "timestamp"
        self.averageRatingField = "average_rating"
        self.ratingNumberField = "rating_number"
        self.priceField = "price"

        # get the dimesnions of the text and image modality encoder embeddings
        textDim = config[self.textField] # 768
        descDim = config[self.descField] # 768
        imgDim = config[self.imgField] # 2048

        # implement layer to learn to map our text and image modality embedding dimensions into the FM model embedding space (embedding_size = 64)
        self.textProjectionLayer = nn.Linear(textDim, self.embSize)
        self.descProjectionLayer = nn.Linear(descDim, self.embSize)
        self.imgProjectionLayer = nn.Linear(imgDim, self.embSize)

        # implement projection layer for the side-feature fields
        #self.ratingProjectionLayer = nn.LazyLinear(embSize)
        self.timestampProjectionLayer = nn.Linear(1, self.embSize)
        self.averageRatingProjectionLayer = nn.Linear(1, self.embSize)
        self.ratingNumberProjectionLayer = nn.Linear(1, self.embSize)
        self.priceProjectionLayer = nn.Linear(1, self.embSize)

        # setup gates for gated sum fusion for all our text and image modality encoder embeddings
        # intention is to apply gates after we have applied the projection layer on our embedding so they are all in a common space
        self.textGate = nn.Linear(self.embSize, self.embSize)
        self.descGate = nn.Linear(self.embSize, self.embSize)
        self.imgGate = nn.Linear(self.embSize, self.embSize)

        # define the layers and loss function
        self.userEmbedding = nn.Embedding(self.numUsers, self.embSize)
        self.itemEmbedding = nn.Embedding(self.numItems, self.embSize)
        self.loss = BPRLoss()
        self.fm = BaseFactorizationMachine(reduce_sum=True)
        self.sigmoid = nn.Sigmoid()
        self.firstOrderLinear = FMFirstOrderLinear(config, dataset)
        self.finalFusionLayer = nn.Linear(self.embSize, 1)

        # initialize parameters in accordance with guidelines for new models in recbole
        self.apply(xavier_normal_initialization)
    

    # create the forward function required to fuse embeddings and feature data into one vector representation
    # will apply gated sum fusion in this section
    def forward(self, interaction):

        # get the user and item ID embeddings
        userID = interaction[self.userIDField]
        itemID = interaction[self.itemIDField]

        #print("INTERACTION: USERID and ITEMID")
        #print("USERID", userID) # [6,6,6,...,19, 19, ... ,25,25...25] -> on GPU
        #print("INTERACTION USER ID SHAPE", userID.shape) # torch.Size([303]) -> an array of 303 elements
        #print("ITEMID", itemID) # [10, 93641, 65184, .... 85676] -> on GPU
        #print("INTERACTION ITEM ID SHAPE", itemID.shape) # torch.Size[303] -> an array of 303 elements

        userIDEmb = self.userEmbedding(userID)
        #print("USER ID EMBEDDING TEST", userIDEmb) # tensor([[ 0.035, -0.072 ,...], [0.0035, -0.072, ...]]) -> on GPU
        #print("USER ID EMB SHAPE", userIDEmb.shape) # torch.Size([303, 64]) -> 64 is what we expect from embedding_size declared in the config

        itemIDEmb = self.itemEmbedding(itemID)
        #print("ITEM ID EMBEDDING TEST", itemIDEmb) # tensor([[ 0.0136, -0.0002, ...], [0.0028, -0.0035, ...]]) -> on GPU
        #print("ITEM ID EMB SHAPE", itemIDEmb.shape)  # torch.Size([303, 64])

        # setup device - confirmed cuda:0
        device = itemIDEmb.device

        # INTERACTION TEST
        #print("INTERACTION: BERT TEXT EMB FIELD FROM .INTER") 
        #print(interaction[self.textField]) # tensor([[-3.3199e-01, -4.8726e-01, ...]]) -> on GPU
        #print("INTERACTION TEXT BERT EMB FROM .INTER SHAPE", interaction[self.textField].shape) # torch.Size[303, 768] -> an array of 303 elements each of size 768 (text emb vector for BERT)

        #print("INTERACTION: BERT DESC EMB FIELD FROM .ITEM")
        #print(interaction[self.descField]) # tensor([[0., 0., .... 0]]) -> on GPU (device='cuda:0')
        #print("INTERACTION DESC BERT EMB FROM .ITEM SHAPE", interaction[self.descField].shape) # torch.Size[303, 768] -> an array of 303 elements each of size 768 (text emb vector for BERT)

        #print("INTERACTION: RESNET IMG EMB FIELD FROM .ITEM")
        #print(interaction[self.imgField]) # tensor([[0.2225, 0.7562, ...], [0.2957, 1.0275, ...]]) ->On GPU -> device='cuda:0
        #print("INTERACTION RESNET IMG EMB FROM .ITEM SHAPE", interaction[self.imgField].shape) # torch.Size[303, 2048] -> an array of 303 elements each of szie 2048 (resnet50 img emb vector size expected)

        # get the precomputted embeddings from the atomic files and apply projection layers to the main text and image modality features
        textEmb = self.textProjectionLayer(interaction[self.textField])
        descEmb = self.descProjectionLayer(interaction[self.descField])
        imgEmb = self.imgProjectionLayer(interaction[self.imgField])

        # apply projections to the side features to convert to FM expected embedding size

        # do not use rating as it is utilized by recbole as a threshold label via the .yaml config
        #ratingEmb = self.ratingProjectionLayer(interaction[self.ratingField])

        #print("TEST STARTS HERE NOW YAY!")

        # side features from .inter

        # TIMESTAMP INTERACTION TEST
        #print("INTERACTION: TIMESTAMP FIELD FROM .INTER") 
        #print(interaction[self.timestampField]) # tensor([1.5497e+12, 1.5497e+12, ... 1.4703e+12]) -> on GPU
        #print("INTERACTION TIMESTAMP FLOAT FIELD FROM .INTER SHAPE", interaction[self.timestampField].shape) # torch.Size([303]) -> this is just 303 elements of timestamp floats in a tensor array

        # unsqueeze to set torch shape to [303, 1] to say we have 303 elements of size 1 before we pass it to the projection layer
        timestampEmb = self.timestampProjectionLayer(interaction[self.timestampField].unsqueeze(-1))


        #print("DID YOU BREAK NOW? 1")

        # side features from .item

        # AVERAGE_RATING INTERACTION TEST
        #print("INTERACTION: AVERAGE_RATING FIELD FROM .ITEM") 
        #print(interaction[self.averageRatingField]) # tensor([3.4000, 3.4000, 4.3000, ... 3.7000]) -> on GPU
        #print("INTERACTION AVERAGE_RATING FLOAT FROM .ITEM SHAPE", interaction[self.averageRatingField].shape) # torch.Size[303] -> this is just 303 elements of average_rating in a tensor array

        # AVERAGE_RATING VIA DATASET TEST
        #print("FROM DATASET TEST: AVERAGE_RATING FIELD FROM .ITEM") 
        #print(self.dataset.item_feat[self.averageRatingField]) # tensor([3.9283, 4.3000, .... ,3.1000, 3.6000, 3.5000]) -> probably on CPU, does not specify any device
        #print("FROM DATASET TEST AVERAGE_RATING FLOAT FROM .ITEM SHAPE", self.dataset.item_feat[self.averageRatingField].shape) #torch.Size[100001] -> this is all instances of average_rating in the .item atomic file provided

        # AVERAGE_RATING VIA DATASET TEST INDEXED BY itemID

        #NOTE: Need to call .to(device) to attempt an index because this self.dataset.item_feat form of access for average_rating is stored on CPU and we need it on GPU to attempt index without error
        #print("FROM DATASET and INDEXED VIA itemID TEST: AVERAGE_RATING FIELD FROM .ITEM") 
        #print(self.dataset.item_feat[self.averageRatingField].to(device)[itemID])
        #print("FROM DATASET and INDEXED VIA itemID TEST AVERAGE_RATING FLOAT FROM .ITEM SHAPE", self.dataset.item_feat[self.averageRatingField].to(device)[itemID].shape)


        # old method
        #averageRatingEmb = self.averageRatingProjectionLayer(self.dataset.item_feat[self.averageRatingField].to(device)[itemID].unsqueeze(-1))
        averageRatingEmb = self.averageRatingProjectionLayer(interaction[self.averageRatingField].unsqueeze(-1))

        # RATING_NUMBER INTERACTION TEST
        #print("INTERACTION: RATING_NUMBER FIELD FROM .ITEM") 
        #print(interaction[self.ratingNumberField]) # tensor([8.0000e+00, 7.0000e+00, ... 4.0000e+00]) -> on GPU
        #print("INTERACTION RATING_NUMBER FLOAT FROM .ITEM SHAPE", interaction[self.ratingNumberField].shape) # torch.Size[303] -> this is 303 elements of different rating numbers in a tensor array

        # old method
        #ratingNumberEmb = self.ratingNumberProjectionLayer(self.dataset.item_feat[self.ratingNumberField].to(device)[itemID].unsqueeze(-1))
        ratingNumberEmb = self.ratingNumberProjectionLayer(interaction[self.ratingNumberField].unsqueeze(-1))

        # PRICE INTERACTION TEST
        #print("INTERACTION: PRICE FIELD FROM .ITEM") 
        #print(interaction[self.priceField]) # tensor([35.7181, 35.7181, ..., 79.9500, ... 35.7181]) -> on GPU
        #print("INTERACTION PRICE FLOAT FROM .ITEM SHAPE", interaction[self.priceField].shape) # torch.Size[303] -> 303 elements of price in a tensor array

        # old method
        #priceEmb = self.priceProjectionLayer(self.dataset.item_feat[self.priceField].to(device)[itemID].unsqueeze(-1))
        priceEmb = self.priceProjectionLayer(interaction[self.priceField].unsqueeze(-1))

        #print("DID YOU BREAK NOW? 2")

        # compute the gates for the text and image modality embeddings
        textGate = self.sigmoid(self.textGate(textEmb))
        descGate = self.sigmoid(self.descGate(descEmb))
        imgGate = self.sigmoid(self.imgGate(imgEmb))

        #print("DID YOU BREAK NOW? 3")

        # apply gated sum fusion on the .inter and .item modality components
        interFusion = userIDEmb + itemIDEmb + timestampEmb + (textGate * textEmb)
        itemFusion = itemIDEmb + averageRatingEmb + ratingNumberEmb + priceEmb + (descGate * descEmb) + (imgGate * imgEmb)

        #print("TEST INTER FUSION RESULT", interFusion) # tensor([[ 4.1741e+11, -9.8166e+10, ...], [4.1741e+11, -9.8166e+10, ...]]) on GPU
        #print("TEST INTER FUSION SHAPE", interFusion.shape) # torch.Size([303, 64]) -> 303 elements, each of size 64

        #print("TEST ITEM FUSION RESULT", itemFusion) # tensor([[-3.0827, -3.9211, ...], [-2.8544, -4.2573, ...]]) # on GPU
        #print("TEST ITEM FUSION SHAPE", itemFusion.shape) # torch.Size([303, 64]) -> 303 elements each of size 64

        completeFusion = interFusion + itemFusion
        #print("TEST COMPLETE FUSION (INTER + ITEM)", completeFusion) # tensor([[ 4.1741e+11, -98166e+10, ..], ..., [3.9602e+11, -9.137e+10, ..]]) on GPU
        #print("TEST COMPLETE FUSION SHAPE", completeFusion.shape) #torch.Size([303, 64]) -> 303 elements each of size 64

        #print("DID YOU BREAK NOW? 4")

        #fmInput = torch.stack([interFusion, itemFusion], dim=1)

        #print("DID IT BREAK NOW? 5")

        # old method imitiated from existing fm.py in recbole library
        # score = self.firstOrderLinear(interaction) + self.fm(fmInput)

        # convert complete fusion from torch.Size([303, 64]) to torch.Size([303, 1])
        # this gets us 303 scores of shape [303], which we need in this function when we are provided 303 user-item pairings in this example forward interaction call
        # batch initially provided in forward function is 303 in this example
        scoresCalculated = self.finalFusionLayer(completeFusion).squeeze(-1)

        #print("WOW WE MADE IT THROUGH!")

        return scoresCalculated


    # from the recbole official docs - used to compute the loss, where the input parameters are Interaction - https://recbole.io/docs/developer_guide/customize_models.html
    # returns a torch.Tensor for computing the BP information
    # implement loss function for pairwise BPR
    def calculate_loss(self, interaction):
        user = interaction[self.userIDField]
        pos_item = interaction[self.itemIDField]
        #print("DID YOU BREAK HERE?")
        neg_item = interaction[self.negItemIDField]
        #print("DID YOU BREAK NOW?")

        user_e = self.userEmbedding(user)                         # [batch_size, embedding_size]
        pos_item_e = self.itemEmbedding(pos_item)                 # [batch_size, embedding_size]
        neg_item_e = self.itemEmbedding(neg_item)                 # [batch_size, embedding_size]
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1) # [batch_size]
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1) # [batch_size]

        loss = self.loss(pos_item_score, neg_item_score)          # []

        return loss
    

    # with full_sort turned off, pass interaction into the forward function
    # this function is based off the recbole fm.py implementation

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
        

    # from the docs - this is an accelerated version of the predict method that allows us to evaluate the full ranking
    # this function is utilized for ranking all items for a given user