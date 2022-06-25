import os
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.utils import (action_on_extraction, form_list_from_user_input,
                         reencode_video_with_diff_fps,
                         show_predictions_on_dataset)

# import traceback
from .randaugment import RandAugmentMC

RESIZE_SIZE = 256
CENTER_CROP_SIZE = 224
TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD = [0.229, 0.224, 0.225]


class ExtractResNet(torch.nn.Module):

    def __init__(self, args):
        super(ExtractResNet, self).__init__()
        self.feature_type = args.feature_type
        self.batch_size = 30
        self.central_crop_size = CENTER_CROP_SIZE
        self.extraction_fps = args.extraction_fps
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CENTER_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
        ])
        self.transforms_weak = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=CENTER_CROP_SIZE,
                              padding=int(CENTER_CROP_SIZE*0.125),
                              padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
        ])
        self.transforms_strong = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=CENTER_CROP_SIZE,
                              padding=int(CENTER_CROP_SIZE*0.125),
                              padding_mode='reflect'),
            RandAugmentMC(n=8, m=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD)
        ])
        self.show_pred = args.show_pred
        # not used, create an issue if you would like to save the frames
        self.keep_tmp_files = args.keep_tmp_files
        self.on_extraction = args.on_extraction
        self.tmp_path = os.path.join(args.tmp_path, self.feature_type)
        self.output_path = os.path.join(args.output_path, self.feature_type)

    def forward(self, indices: torch.LongTensor):
        '''
        Arguments:
            indices {torch.LongTensor} -- indices to self.path_list
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, class_head = self.load_model(device)


        try:
            feat = self.extract(device, model, class_head, indices)
            # value = action_on_extraction(feats_dict, indices, self.output_path, self.on_extraction)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            # prints only the last line of an error. Use `traceback.print_exc()` for the whole traceback
            print(e)
            print(f'Extraction failed at: {indices} with error (↑). Continuing extraction')
        return feat

    def extract(self, device: torch.device, model: torch.nn.Module, classifier: torch.nn.Module,
                video_path: Union[str, None] = None) -> Dict[str, np.ndarray]:
        '''The extraction call. Made to clean the forward call a bit.

        Arguments:
            device {torch.device}
            model {torch.nn.Module}
            classifier {torch.nn.Module} -- pre-trained classification layer, will be used if
                                            show_pred is True

        Keyword Arguments:
            video_path {Union[str, None]} -- if you would like to use import it and use it as
                                             "path -> model"-fashion (default: {None})

        Returns:
            Dict[str, np.ndarray]: 'features_nme', 'fps', 'timestamps_ms'
        '''
        def _run_on_a_batch(vid_feats, batch, model, classifier, device):
            batch = torch.cat(batch).to(device)

            with torch.no_grad():
                batch_feats = model(batch)
                vid_feats.extend(batch_feats.tolist())
                # show predicitons on imagenet dataset (might be useful for debugging)
                if self.show_pred:
                    logits = classifier(batch_feats)
                    show_predictions_on_dataset(logits, 'imagenet')

        # take the video, change fps and save to the tmp folder
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        print(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamps_ms = []
        batch = []
        vid_feats = []
        batch_weak = []
        batch_strong = []
        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True
        cap.set(1,270)
        while cap.isOpened():
            frame_exists, rgb = cap.read()

            if first_frame:
                first_frame = False
                if frame_exists is False:
                    continue

            if frame_exists:
                timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                # prepare data (first -- transform, then -- unsqueeze)
                # cv2.imwrite("1.jpeg", np.array(rgb))
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = self.transforms(rgb)
                rgb_weak = self.transforms_weak(rgb)
                rgb_strong = self.transforms_strong(rgb)
                # print(cv2.imwrite("2.jpeg", np.array(rgb).reshape(224,224,3)))
                rgb = rgb.unsqueeze(0)
                rgb_weak = rgb_weak.unsqueeze(0)
                rgb_strong = rgb_strong.unsqueeze(0)
                batch.append(rgb)
                batch_weak.append(rgb_weak)
                batch_strong.append(rgb_strong)
                # when batch is formed to inference
                if len(batch) == self.batch_size:
                    batch+=batch_weak
                    batch+=batch_strong
                    _run_on_a_batch(vid_feats, batch, model, classifier, device)
                    # clean up the batch list
                    batch = []
            else:
                # if the last batch was smaller than the batch size
                if len(batch) != 0:
                    batch.append(batch_weak)
                    batch.append(batch_strong)
                    _run_on_a_batch(vid_feats, batch, model, classifier, device)
                
                cap.release()
                break

        # removes the video with different fps if it was created to preserve disk space
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            os.remove(video_path)

        # features_with_meta = {
        #     self.feature_type: np.array(vid_feats),
        #     'fps': np.array(fps),
        #     'timestamps_ms': np.array(timestamps_ms)
        # }

        return vid_feats

    def load_model(self, device: torch.device) -> Tuple[torch.nn.Module]:
        '''Defines the models, loads checkpoints, sends them to the device.

        Args:
            device (torch.device): The device

        Raises:
            NotImplementedError: if flow type is not implemented.

        Returns:
            Tuple[torch.nn.Module]: the model with identity head, the original classifier
        '''
        if self.feature_type == 'resnet18':
            model = models.resnet18
        elif self.feature_type == 'resnet34':
            model = models.resnet34
        elif self.feature_type == 'resnet50':
            model = models.resnet50
        elif self.feature_type == 'resnet101':
            model = models.resnet101
        elif self.feature_type == 'resnet152':
            model = models.resnet152
        else:
            raise NotImplementedError

        model = model(pretrained=True)
        model = model.to(device)
        model.eval()
        # save the pre-trained classifier for show_preds and replace it in the net with identity
        class_head = model.fc
        model.fc = torch.nn.Identity()
        return model, class_head
