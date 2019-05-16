from pathlib import Path
from Experiment import Experiment

# Set the path to load data
data_path = Path('../data/pickles')
# Set the path to save predictions
logs_path = Path('../logs')
# Set the path to save models
models_path = Path('../models')

# Modify the hyperparamters as needed
learning_rate = 1e-5
epochs = 100
batch_size = 128
image_size = 227
comet_api = ''

# Create an experiment object
goturn_exp = Experiment(learning_rate=learning_rate,
                        image_size=image_size,
                        data_path=data_path,
                        augment=False,
                        logs_path=logs_path,
                        models_path=models_path,
                        comet_api=comet_api)

# Train the model
losses = goturn_exp.train(epochs, batch_size, validate=True, test=True)
