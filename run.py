#from recbole.quick_start import run_recbole

#run_recbole(model='BPR', dataset='ml-100k')

# next step to load dataset - https://recbole.io/docs/user_guide/usage/running_new_dataset.html

# GOAL: Run baseline FM multimodal recommender against our generated atomic files with text (BERT) and image embeddings (RESNET50)

from logging import getLogger

# override PyTorch 2.6+ safety flag by forcing weights_only=False - will not affect model accuracy just remove security for checkpoints
import torch, functools
torch.load = functools.partial(torch.load, weights_only=False)


from recbole.utils import init_seed, init_logger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.model.context_aware_recommender import FM

# custom model
#from recbole.model.custom_recommender.gatedsumfusion import GatedSumFusion
from recbole.model.custom_recommender.finegrainedfusion import FineGrainedFusion


# based on recbole official docs for running a custom model https://recbole.io/docs/developer_guide/customize_models.html
if __name__ == "__main__":
    # load configuration with FM model with Amazon_Fashion dataset
    config = Config(
        model=FineGrainedFusion,
        config_file_list=['C:/Users/ronni/OneDrive/Desktop/Optimal-MRS-Project/configs/BERT_RES50_Amazon_Fashion.yaml']
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

    # initialize the baseline FM model with the config file and training data - also move to GPU device or CPU if this is not available

    #baselineModel = FM(config, train_data.dataset).to(config['device'])

    baselineModel = FineGrainedFusion(config, train_data.dataset).to(config['device'])

    # setup the trainer for the baseline model
    trainer = Trainer(config, baselineModel)

    # fit the trainer with the training and validation dataset and get the best valid score and result
    bestValidScore, bestValidResult = trainer.fit(train_data, valid_data)

    # evaluate the test data against the model and display best results found
    testResult = trainer.evaluate(test_data)
    logger.info(f"Best validation metric score found across all epochs: {bestValidScore}")
    logger.info(f"All metrics specified from config from where the best validation metric score took place: {bestValidResult}")

    # log result of model against unseen test data
    logger.info(f"Test result for our model: {testResult}")