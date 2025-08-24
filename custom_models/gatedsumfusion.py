import torch
from torch import nn

from recbole.utils import InputType
from recbole.model.loss import BPRLoss
from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import ContextRecommender

# NOTE:
# the following gated-sum fusion model implementation is inspired and adapted from the following paper:

'''
Liu, S., Zhang, Y., Li, X., Liu, Y., Feng, C., & Yang, H. (2025). Gated Multimodal Graph
Learning for Personalized Recommendation (arXiv:2506.00107). arXiv.
https://doi.org/10.48550/arXiv.2506.00107

'''

# gated sum fusion is only performed on text and image modality encoder embeddings
# all other features such as average_rating can be projected and concatenated without a gate
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
        self.textField = config["text_field"]
        self.descField = config["desc_field"]
        self.imgField = config["img_field"]

        # get the side-feature fields
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
        self.timestampProjectionLayer = nn.Linear(1, self.embSize)
        self.averageRatingProjectionLayer = nn.Linear(1, self.embSize)
        self.ratingNumberProjectionLayer = nn.Linear(1, self.embSize)
        self.priceProjectionLayer = nn.Linear(1, self.embSize)

        # setup gates for gated sum fusion for all our text and image modality encoder embeddings
        # intention is to apply gates after we have applied the projection layer on our embedding so they are all in a common space
        self.scalarGate = nn.Linear(self.embSize * 3, 1)

        # define the layers and loss function
        self.userEmbedding = nn.Embedding(self.numUsers, self.embSize)
        self.itemEmbedding = nn.Embedding(self.numItems, self.embSize)
        self.loss = BPRLoss()
        self.sigmoid = nn.Sigmoid()
        self.finalFusionLayer = nn.Linear(self.embSize, 1)

        # initialize parameters in accordance with guidelines for new models in recbole
        self.apply(xavier_normal_initialization)
    

    # create the forward function required to fuse embeddings and feature data into one vector representation
    # will apply gated sum fusion in this section
    def forward(self, interaction):

        # get the user and item ID embeddings
        userID = interaction[self.userIDField]
        itemID = interaction[self.itemIDField]

        userIDEmb = self.userEmbedding(userID)
        itemIDEmb = self.itemEmbedding(itemID)
        
        # get the precomputted embeddings from the atomic files and apply projection layers to the main text and image modality features
        textEmb = self.textProjectionLayer(interaction[self.textField])
        descEmb = self.descProjectionLayer(interaction[self.descField])
        imgEmb = self.imgProjectionLayer(interaction[self.imgField])

        # apply projections to the side features to convert to FM expected embedding size
        # do not use rating as it is utilized by recbole as a threshold label via the .yaml config

        # unsqueeze to set torch shape to [303, 1] to say we have 303 elements of size 1 before we pass it to the projection layer
        timestampEmb = self.timestampProjectionLayer(interaction[self.timestampField].unsqueeze(-1))
        averageRatingEmb = self.averageRatingProjectionLayer(interaction[self.averageRatingField].unsqueeze(-1))
        ratingNumberEmb = self.ratingNumberProjectionLayer(interaction[self.ratingNumberField].unsqueeze(-1))
        priceEmb = self.priceProjectionLayer(interaction[self.priceField].unsqueeze(-1))

        # compute the gates for the text and image modality embeddings
        finalGate = self.sigmoid(self.scalarGate(torch.cat([textEmb, descEmb, imgEmb], dim=-1)))

        # apply gated sum fusion on the .inter and .item modality components
        interFusion = userIDEmb + itemIDEmb + timestampEmb + ((1- finalGate) * textEmb)
        itemFusion = itemIDEmb + averageRatingEmb + ratingNumberEmb + priceEmb + ((1- finalGate) * descEmb) + (finalGate * imgEmb)

        completeFusion = interFusion + itemFusion

        # convert complete fusion from torch.Size([303, 64]) to torch.Size([303, 1])
        # this gets us 303 scores of shape [303], which we need in this function when we are provided 303 user-item pairings in this example forward interaction call
        # batch initially provided in forward function is 303 in this example
        scoresCalculated = self.finalFusionLayer(completeFusion).squeeze(-1)
        return scoresCalculated


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