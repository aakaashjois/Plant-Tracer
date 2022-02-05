from torch.utils.data.dataset import Dataset
from xml.etree import ElementTree
from zipfile import ZipFile
from io import BytesIO
import numpy as np
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
from albumentations import HorizontalFlip, ShiftScaleRotate, RandomBrightnessContrast, Compose


class Data(Dataset):
    """
    A PyTorch Dataset class to load and process the Plant Tracer data

    Parameters
    ----------
    path: Path
        Path to location of frames and annotations
        The videos are expected as ZIP file containing all the frames
        The annotations are expected in PASCAL_VOC format
    target_size: int
        The size of image to use when training the model
        The image will be of size (@target_size x @target_size)
    transforms: bool
        Apply augmentation transformations on the image
    """

    def __init__(self, path: Path, target_size: int, transforms: bool = True):
        self.target_size = target_size
        self.transforms = transforms
        # Load the pickle files if the path contains "pickle"
        # Pickles are much faster to load and makes debugging easier
        if 'pickle' in str(path.resolve()):
            self.video_annotations, self.video_frames = self.load_pickles(path)
        else:
            all_frames = list(path.glob('*.zip'))
            all_annotations = list(path.glob('*.xml'))
            self.video_annotations = [self.load_annotations(a) for a in
                                      tqdm(all_annotations, desc='Loading Annotations')]
            self.video_frames = [self.load_frames(f) for f in tqdm(all_frames, desc='Loading Video Frames')]
        self.__sanity_check__()
        if self.transforms:
            self.prev_hflip = Compose([HorizontalFlip(p=1)])
            self.prev_ssr = Compose([ShiftScaleRotate(p=1)])
            self.prev_bri = Compose([RandomBrightnessContrast(p=1)])
            self.curr_hflip = Compose([HorizontalFlip(p=1)],
                                      bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']})
            self.curr_ssr = Compose([ShiftScaleRotate(p=1)],
                                    bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']})
            self.curr_bri = Compose([RandomBrightnessContrast(p=1)],
                                    bbox_params={'format': 'pascal_voc', 'label_fields': ['category_id']})

    def __len__(self):
        """
        Calculate the total length of dataset
        Returns
        -------
            int: Length of dataset
        """
        return sum(len(videos) - 1 for videos in self.video_frames)

    def __getitem__(self, index):
        """
        Retrieve a random item from the dataset

        Returns
        -------
            ndarray: The previous frame cropped based on annotation
            ndarray: The current frame cropped based on previous annotation
            ndarray: The current annotation rescaled
            list: The scale used to crop resize the frames
            list: The amount of crop performed on the frames
        """
        # Select a random video and then, select a random frame in that video
        np.random.seed(seed=index)
        vi = np.random.randint(low=0, high=len(self.video_frames) - 1, size=1)[0]
        fi = np.random.randint(low=1, high=len(self.video_frames[vi]), size=1)[0]
        np.random.seed(seed=None)
        previous_frame = self.video_frames[vi][fi - 1]
        current_frame = self.video_frames[vi][fi]
        previous_annotation = self.video_annotations[vi][fi - 1]
        current_annotation = self.video_annotations[vi][fi]
        return self.make_crops(previous_frame, current_frame, previous_annotation, current_annotation)

    def __sanity_check__(self):
        """
        Make a check to see if number of videos and annotations match
        Raises
        -------
        ValueError
            If the number of video frames does not match the number of annotations, or
            if each video does not have its respective annotations
        """
        if len(self.video_annotations) != len(self.video_frames):
            raise ValueError('Sizes of annotations and videos do not match')
        for i, d in enumerate(zip(self.video_annotations, self.video_frames)):
            if d[0].shape[0] != d[1].shape[0]:
                raise ValueError('Sizes of annotations and videos do not match in {}'.format(i + 1))

    @staticmethod
    def load_pickles(path: Path):
        """
        Load the video frames and annotations using pickle files

        Parameters
        ----------
        path: Path
            Path to location of pickle file which contains the video frames and annotations

        Returns
        -------
            Video annotations and video frames extracted from the pickle file
        """
        pickles = list(path.glob('*.pkl'))
        video_annotations = []
        video_frames = []
        for p in tqdm(pickles, desc='Loading Pickles'):
            with open(p, 'rb') as pkl:
                save_dict = pickle.load(pkl)
                video_annotations.append(save_dict['annotations'])
                video_frames.append(save_dict['frames'])
        return video_annotations, video_frames

    @staticmethod
    def load_annotations(path: Path):
        """
        Load the annotations from XML file
        Parameters
        ----------
        path: Path
            Path pointing to the XML file which contains the annotations
        Returns
        -------
            ndarray: The annotations loaded as an array
        """
        #
        root = ElementTree.parse(path).getroot()
        polygons = root[3].findall('polygon')
        buffer = np.empty((len(polygons), 4), np.int)
        for i, polygon in enumerate(polygons):
            pts = polygon.findall('pt')
            buffer[i][0] = int(pts[1][0].text)  # top-left x
            buffer[i][1] = int(pts[1][1].text)  # top-left y
            buffer[i][2] = int(pts[3][0].text)  # bottom-right x
            buffer[i][3] = int(pts[3][1].text)  # bottom-right y
        return buffer

    @staticmethod
    def load_frames(path: Path):
        """
        Load the frames from the ZIP file
        Parameters
        ----------
        path: Path
            Path pointing to the ZIP file which contains the video frames
        Returns
        -------
            ndarray: The video frames loaded as an array
        """
        zip_file = ZipFile(path)
        names = zip_file.namelist()
        buf = [cv2.imdecode(np.frombuffer(BytesIO(zip_file.open(name).read()).read(), np.uint8), 1) for name in names]
        return np.array(buf)

    @staticmethod
    def load_video(path: Path):
        """
        Load video frames from video file
        Parameters
        ----------
        path: Path
            Path pointing to the video file which has to be loaded
        Returns
        -------
            ndarray: The video frames loaded as an array
        """
        #
        cap = cv2.VideoCapture(str(path.resolve()))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buffer = np.empty((frame_count, frame_height, frame_width, 3), np.uint8)
        fc = 0
        ret = True
        while fc < frame_count and ret:
            ret, buffer[fc] = cap.read()
            fc += 1
        cap.release()
        return buffer

    @staticmethod
    def convert_channels(image: np.ndarray, channel_first: bool = False, channel_last: bool = False):
        """
        Converts image to channel first or channel last format

        Parameters
        ----------
        image: ndarray
            The image array
        channel_first: bool
            Set this to True to convert from channel last to channel first
        channel_last: bool
            Set this to True to convert from channel first to channel last

        Returns
        -------
            ndarray: Channel converted image
        """
        return np.moveaxis(image, -1, 0) if channel_first else (np.moveaxis(image, 0, -1) if channel_last else image)

    @staticmethod
    def get_bbox(preds: np.ndarray, target_size: int, scale: list, crop: list):
        """
        Recovers the bounding boxes after prediction

        Parameters
        ----------
        preds: ndarray
            The predicted bounding box coordinates
        target_size: int
            The image size
        scale: list
            The scale used when rescaling the bounding boxes
        crop: list
            The crop used when resizing the bounding boxes
        Returns
        -------
            ndarray: Recovered bounding box
        """
        preds = np.divide(preds, 10)
        preds = np.multiply(preds, target_size)
        preds = np.divide(preds, scale * 2)
        preds = np.add(preds, crop * 2)
        return preds.astype(int)

    def make_crops(self, previous_frame: np.ndarray,
                   current_frame: np.ndarray,
                   previous_annotation: np.ndarray,
                   current_annotation: np.ndarray,
                   validate: bool = False):
        """

        Parameters
        ----------
        previous_frame: ndarray
            The previous frame of the selected video frame
        current_frame: ndarray
            The current video frame
        previous_annotation: ndarray
            The annotation for the previous video frame
        current_annotation: ndarray
            The annotation for the current video frame
        validate: bool, default = False
            Will not perform image transformation if set to True
        Returns
        -------
            ndarray: The previous frame cropped based on annotation
            ndarray: The current frame cropped based on previous annotation
            ndarray: The current annotation rescaled
            list: The scale used to crop resize the frames
            list: The amount of crop performed on the frames
        """
        # Clip the bounding box to makes it lies inside the image
        previous_annotation[0] = np.clip(previous_annotation[0], 0, previous_frame.shape[0])
        previous_annotation[1] = np.clip(previous_annotation[1], 0, previous_frame.shape[1])
        previous_annotation[2] = np.clip(previous_annotation[2], 0, previous_frame.shape[0])
        previous_annotation[3] = np.clip(previous_annotation[3], 0, previous_frame.shape[1])

        center_x = int((previous_annotation[0] + previous_annotation[2]) / 2)
        center_y = int((previous_annotation[1] + previous_annotation[3]) / 2)
        width = abs(previous_annotation[2] - previous_annotation[0])
        height = abs(previous_annotation[3] - previous_annotation[1])

        # Create a crop window of size 7 x max(height, width) of the image
        crop_size = np.clip(7 * max(width, height), 10, 120)
        top = np.clip(center_x + crop_size, 0, previous_frame.shape[0])
        left = np.clip(center_y - crop_size, 0, previous_frame.shape[1])
        bottom = np.clip(center_x - crop_size, 0, previous_frame.shape[0])
        right = np.clip(center_y + crop_size, 0, previous_frame.shape[1])
        crop = [bottom, left]

        # Generate the cropped images
        previous_cropped = previous_frame[left: right, bottom: top, :]
        current_cropped = current_frame[left: right, bottom: top, :]

        # Calculate the scale needed to resize the image to :target_size:
        scale = np.divide(self.target_size, current_cropped.shape[:-1]).tolist()
        previous_cropped = cv2.resize(previous_cropped, (self.target_size, self.target_size))
        current_cropped = cv2.resize(current_cropped, (self.target_size, self.target_size))

        # Scale and crop the bounding box appropriately
        bbox = np.subtract(current_annotation, crop * 2)
        bbox = np.multiply(bbox, scale * 2)
        bbox = np.divide(bbox, self.target_size)

        # Apply transformations
        if not validate and self.transforms:
            try:
                x_min, y_max, x_max, y_min = bbox
                bbox = np.array([x_min, y_min, x_max, y_max])
                previous_augmented = {'image': previous_cropped}
                current_augmented = {'image': current_cropped, 'bboxes': [bbox], 'category_id': [0]}
                if np.random.random() > 0.5:
                    try:
                        previous_augmented = self.prev_hflip(**previous_augmented)
                        current_augmented = self.curr_hflip(**current_augmented)
                    except Exception as e:
                        print(e)
                if np.random.random() > 0.5:
                    try:
                        previous_augmented = self.prev_ssr(**previous_augmented)
                        current_augmented = self.curr_ssr(**current_augmented)
                    except Exception as e:
                        print(e)
                if np.random.random() > 0.5:
                    try:
                        previous_augmented = self.prev_bri(**previous_augmented)
                        current_augmented = self.curr_bri(**current_augmented)
                    except Exception as e:
                        print(e)
                previous_cropped = previous_augmented['image']
                current_cropped = current_augmented['image']
                x_min, y_min, x_max, y_max = bbox
                bbox = np.array([x_min, y_max, x_max, y_min])
            except Exception as e:
                print(e)

        # Convert images from channel last to channel first format
        previous_cropped = self.convert_channels(previous_cropped, channel_first=True)
        current_cropped = self.convert_channels(current_cropped, channel_first=True)

        # Multiply the bounding box by 10
        bbox = np.multiply(bbox, 10)
        return previous_cropped, current_cropped, bbox, scale, crop
