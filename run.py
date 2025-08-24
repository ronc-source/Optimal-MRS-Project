from logging import getLogger
import time

# override PyTorch 2.6+ safety flag by forcing weights_only=False - will not affect model accuracy just remove security for checkpoints
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)

from recbole.utils import init_seed, init_logger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer

# baseline from RecBole
from recbole.model.context_aware_recommender import FM

# near quality to SOTA models from RecBole
from recbole.model.context_aware_recommender import xDeepFM
from recbole.model.context_aware_recommender import FiGNN

# custom models
from recbole.model.custom_recommender.gatedsumfusion import GatedSumFusion
from recbole.model.custom_recommender.finegrainedfusion import FineGrainedFusion

# setup main function based on RecBole official docs for running a custom model https://recbole.io/docs/developer_guide/customize_models.html
if __name__ == "__main__":

    # load configuration settings with declared model and reference to .YAML config file
    config = Config(
        model=FineGrainedFusion,
        config_file_list=['C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project/configs/TinyBERT_VITSMALL_Amazon_Fashion.yaml']
    )

    # initailize reproducibility parameters from config file
    init_seed(config['seed'], config['reproducibility'])

    # setup logger and display configuration data such as file paths and hyperparameters in the logs
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # read in the data from the atomic files along with config specifications and display this data in the logs
    dataset = create_dataset(config)

    logger.info(dataset)

    # split data into training, validation and test datasets based on config file specifications
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # NOTE:
    # there are a total of 38,730 positive interactions registered from 2,035,491 users, 100,001 items and an initial .inter file of 200,001
    # the rest of the information provided from the atomic files will be used for negative sampling against each of the positive interactions in train and eval stages
    testDatasetSize = len(test_data.dataset)

    # initialize the selected model with the config file and training data - also move to GPU device or CPU if this is not available
    selectedModel = FineGrainedFusion(config, train_data.dataset).to(config['device'])

    # setup the trainer for the selected model
    trainer = Trainer(config, selectedModel)

    # fit the trainer with the training and validation dataset and get both the best validation score and best validation result found during validation stages in training
    bestValidScore, bestValidResult = trainer.fit(train_data, valid_data)

    # display the best valid_metric, specified in the config file (NDCG@10), for the best model found during training
    logger.info(f"Best model found across all epochs based on valid_metric: {bestValidScore}")

    # display a list of metrics specified in the config file (recall@10, mrr@10, ndcg@10, gauc) for the best model found during training
    logger.info(f"All metrics specified from config for the best model found: {bestValidResult}")
    
    # evaluate the test dataset against the best model found during training and display best results found and record inference time for the model in test set
    startTestTime = time.perf_counter()

    testResult = trainer.evaluate(test_data)

    endTestTime = time.perf_counter()

    # log result of model against unseen test data
    logger.info(f"Test result for our best model found: {testResult}")

    # log average inference time for model against each test dataset row and convert from seconds to ms
    logger.info(f"Average inference time per data row in test set for best model: {(((endTestTime - startTestTime) / testDatasetSize) * 1000)} ms")
    
    # log total time (in seconds) it took our best model to go through the test set
    logger.info(f"Total time it took our best model to go through test dataset: {endTestTime - startTestTime} seconds")