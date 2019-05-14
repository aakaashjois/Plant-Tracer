from comet_ml import Experiment as Comet
import torch
from tqdm import tqdm
import gc
import datetime
import numpy as np
from GoTurnRemix import GoTurnRemix
from Data import Data
from pathlib import Path


class Experiment:
    """
        A helper class to facilitate the training and validation procedure of the GoTurnRemix model

        Parameters
        ----------
        learning_rate: float
            Learning rate to train the model. The optimizer is SGD and the loss is L1 Loss
        image_size: int
            The size of the input image. This has to be fixed before the data is created
        data_path: Path
            Path to the data folder. If the folder name includes "pickle", then the data saved as pickles are loaded
        augment: bool
            Perform augmentation on the images before training
        logs_path: Path
            Path to save the validation predictions at the end of each epoch
        models_path: Path
            Path to save the model state at the end of each epoch
        save_name: str
            Name of the folder in which the logs and models are saved. If not provided, the current datetime is used
    """

    def __init__(self,
                 learning_rate: float,
                 image_size: int,
                 data_path: Path,
                 augment: bool = True,
                 logs_path: Path = None,
                 models_path: Path = None,
                 save_name: str = None,
                 comet_api: str = None):
        self.image_size = image_size
        self.logs_path = logs_path
        self.models_path = models_path
        self.model = GoTurnRemix()
        self.model.cuda()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)
        self.model_name = str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-')
        self.model_name = save_name if save_name else self.model_name
        self.augment = augment
        self.data = Data(data_path, target_size=self.image_size, transforms=augment)
        self.comet = None
        if comet_api:
            self.comet = Comet(api_key=comet_api, workspace="aakaashjois")
            self.comet.log_parameter('learning_rate', learning_rate)
            self.comet.log_parameter('image_size', image_size)
            self.comet.log_parameter('augment', augment)

    def __train_step__(self, data):
        """
        Performs one step of the training procedure

        Parameters
        ----------
        data
            data obtained from @Data.__getitem__

        Returns
        -------
           Loss at the end of training step
        """
        if self.comet:
            self.comet.train()
        previous_cropped, current_cropped, bbox, scale, crop = data
        previous_cropped = torch.div(previous_cropped, 255).float().cuda()
        current_cropped = torch.div(current_cropped, 255).float().cuda()
        previous_cropped = torch.autograd.Variable(previous_cropped, requires_grad=True)
        current_cropped = torch.autograd.Variable(current_cropped, requires_grad=True)
        bbox = bbox.requires_grad_(True).float().cuda()
        self.optimizer.zero_grad()
        preds = self.model(previous_cropped, current_cropped)

        del previous_cropped
        del current_cropped
        gc.collect()

        loss = self.criterion(preds, bbox)
        if self.comet:
            self.comet.log_metric('loss', loss)
        loss.backward()
        self.optimizer.step()
        return loss

    def __test__(self):
        """
        Test tracking of the model

        Returns
        -------
            Test loss and test predictions
        """
        # Set model to evaluation mode
        if self.comet:
            self.comet.test()
        self.model.eval()
        test_preds = []
        test_loss = []
        video_frames = self.data.video_frames[-1]
        video_annotations = self.data.video_annotations[-1]
        p_a = video_annotations[0]
        p_f = video_frames[0]
        test_preds.append(p_a)

        for i in tqdm(range(1, len(video_annotations)), desc='Validating'):
            c_a = video_annotations[i]
            c_f = video_frames[i]
            p_c, c_c, bbox, scale, crop = self.data.make_crops(p_f, c_f, p_a, c_a)
            p_c = torch.div(torch.from_numpy(p_c), 255).unsqueeze(0).float().cuda()
            c_c = torch.div(torch.from_numpy(c_c), 255).unsqueeze(0).float().cuda()
            bbox = torch.tensor(bbox, requires_grad=False).float().cuda()
            preds = self.model(p_c, c_c)

            del p_c
            del c_c
            gc.collect()

            loss = torch.nn.functional.l1_loss(preds, bbox)
            if self.comet:
                self.comet.log_metric('val_loss', loss)
            test_loss.append(loss.item())
            preds = self.data.get_bbox(preds.cpu().detach().numpy()[0], self.image_size, scale, crop)
            test_preds.append(preds)
            p_a = preds
            p_f = c_f
        return test_loss, test_preds

    def __validate__(self):
        """
        Performs validation on the model

        Returns
        -------
            Validation loss and validation predictions
        """
        # Set model to evaluation mode
        if self.comet:
            self.comet.validate()
        self.model.eval()
        validation_preds = []
        validation_loss = []
        video_frames = self.data.video_frames[-1]
        video_annotations = self.data.video_annotations[-1]
        p_a = video_annotations[0]
        p_f = video_frames[0]
        validation_preds.append(p_a)

        for i in tqdm(range(1, len(video_annotations)), desc='Validating'):
            c_a = video_annotations[i]
            c_f = video_frames[i]
            p_c, c_c, bbox, scale, crop = self.data.make_crops(p_f, c_f, p_a, c_a)
            p_c = torch.div(torch.from_numpy(p_c), 255).unsqueeze(0).float().cuda()
            c_c = torch.div(torch.from_numpy(c_c), 255).unsqueeze(0).float().cuda()
            bbox = torch.tensor(bbox, requires_grad=False).float().cuda()
            preds = self.model(p_c, c_c)

            del p_c
            del c_c
            gc.collect()

            loss = torch.nn.functional.l1_loss(preds, bbox)
            if self.comet:
                self.comet.log_metric('val_loss', loss)
            validation_loss.append(loss.item())
            preds = self.data.get_bbox(preds.cpu().detach().numpy()[0], self.image_size, scale, crop)
            validation_preds.append(preds)
            p_a = c_a
            p_f = c_f
        return validation_loss, validation_preds

    def train(self, epochs: int, batch_size: int, validate: bool = True, test: bool = True):
        """
        Trains the model for @epochs number of epochs

        Parameters
        ----------
        epochs: int
            Number of epochs to train the model
        batch_size: int
            The size of each batch when training the model
        validate: bool, default=True
            If True, validation occurs at the end of each epoch
            The results are saved in @logs_path and models are saved in @models_path
        test: bool, default=True
            If True, the model is tested for tracking at the end of the training procedure
            The results are saved in @logs_path

        Returns
        -------
            list: List containing the training loss at the end of each epoch
        """
        if self.comet:
            self.comet.log_parameter('epochs', epochs)
            self.comet.log_parameter('batch_size', batch_size)
        loss_per_epoch = []
        preds_per_epoch = []
        # Run the model without any training to get a baseline prediction
        if validate:
            validation_loss, validation_preds = self.__validate__()
            preds_per_epoch.append(validation_preds)
        # Set the model to training mode
        self.model.train()
        # Create a DataLoader to feed data to the model
        dataloader = torch.utils.data.DataLoader(dataset=self.data, batch_size=batch_size, shuffle=True)

        # Run for @epochs number of epochs
        for epoch in range(epochs):
            if self.comet:
                self.comet.log_current_epoch(epoch)
            running_loss = []
            for step, data in enumerate(tqdm(dataloader,
                                             total=int(len(self.data) / batch_size),
                                             desc='Epoch {}'.format(epoch))):
                loss = self.__train_step__(data)
                running_loss.append(loss.item())
            training_loss = sum(running_loss) / len(running_loss)
            loss_per_epoch.append(sum(running_loss) / len(running_loss))
            if validate:
                validation_loss, validation_preds = self.__validate__()
                preds_per_epoch.append(validation_preds)
                print('Validation loss: {}'.format(sum(validation_loss) / len(validation_loss)))
            # Save the model at this stage
            if self.models_path:
                (self.models_path / self.model_name).mkdir(exist_ok=True)
                torch.save(self.model, (self.models_path / self.model_name / 'epoch_{}'.format(epoch)).resolve())
            print('Training Loss: {}'.format(training_loss))
        # Save the validation frames, ground truths and predictions at this stage
        if self.logs_path:
            (self.logs_path / self.model_name).mkdir(exist_ok=True)
            save = {'frames': self.data.video_frames[-1],
                    'truth': self.data.video_annotations[-1],
                    'preds': preds_per_epoch}
            np.save(str((self.logs_path / self.model_name / 'preds_per_epoch.npy').resolve()), save)
        # Test the model and save the results
        if test:
            test_loss, test_preds = self.__test__()
            if self.logs_path:
                (self.logs_path / self.model_name).mkdir(exist_ok=True)
                save = {'frames': self.data.video_frames[-1],
                        'truth': self.data.video_annotations[-1],
                        'preds': test_preds,
                        'loss': test_loss}
                np.save(str((self.logs_path / self.model_name / 'test_preds.npy').resolve()), save)
        return loss_per_epoch
