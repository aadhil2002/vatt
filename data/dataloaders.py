# vatt/data/dataloaders.py

import functools
from typing import Any, Optional

import tensorflow as tf

from vatt.data import loading
from vatt.data import processing
from vatt.data.datasets import factory as ds_fctr

FeatureNames = processing.FeatureNames

# Constants
DEFAULT_FPS = 25
DEFAULT_SR = 48000
REF_FPS = 10
REF_SR = 48000

SELF_SUP_DS = ['dmvr', 'audioset']
VID_CLS_DS = ['kinetics400', 'kinetics600', 'kinetics700', 'mit', 'hmdb51', 'ucf101']
AUD_CLS_DS = ['audioset', 'esc50']
IMG_CLS_DS = ['imagenet']
CLS_DS = {
    'hmdb51': {'num_classes': 51, 'splits': [1, 2, 3], 'total_steps': 10000},
    'ucf101': {'num_classes': 101, 'splits': [1, 2, 3], 'total_steps': 10000},
    'esc50': {'num_classes': 50, 'splits': [1, 2, 3, 4, 5], 'total_steps': 10000},
    'kinetics400': {'num_classes': 400},
    'kinetics600': {'num_classes': 600},
    'kinetics700': {'num_classes': 700},
    'mit': {'num_classes': 339},
    'imagenet': {'num_classes': 1000},
    'audioset': {'num_classes': 527},
}
TEXT_DS = {'dmvr': {'num_clips': 10000}, 'youcook2': {'num_clips': 3320}, 'msrvtt': {'num_clips': 1000}, 'msrvtt1000': {'num_clips': 1000}}
AUDIO_DS = ['audioset', 'esc50']

class PreTrainLoader(loading.BaseLoader):
    """Constructs the dataloader for pre-training."""

    def __init__(self, dataset_id, params):
        # Generic parameters
        input_params = params.train.input
        self._num_frames = input_params.num_frames
        self._frame_size = input_params.frame_size
        self._video_stride = input_params.video_stride
        self._raw_audio = input_params.raw_audio
        self._stft_length = input_params.stft_length
        self._stft_step = input_params.stft_step
        self._mel_bins = input_params.mel_bins
        self._zero_centering_image = input_params.zero_centering_image
        self._max_num_words = input_params.max_num_words
        self._max_context_sentences = input_params.max_context_sentences
        self._space_to_depth = input_params.space_to_depth
        self._linearize_vision = input_params.linearize_vision
        self.data_path = input_params.data_path

        # Augmentation parameters
        self._min_resize = input_params.min_resize
        self._min_area_ratio = input_params.min_area_ratio
        self._max_area_ratio = input_params.max_area_ratio
        self._min_aspect_ratio = input_params.min_aspect_ratio
        self._max_aspect_ratio = input_params.max_aspect_ratio
        self._crop_resize_style = input_params.crop_resize_style
        self._scale_jitter = input_params.scale_jitter
        self._audio_noise = input_params.audio_noise
        self._audio_mixup = input_params.audio_mixup
        self._mixup_alpha = input_params.mixup_alpha
        self._mixup_beta = input_params.mixup_beta

        ds_names = dataset_id.split('+')
        ds_factories = []
        for ds_name in ds_names:
            params_factory = {
                'is_training': True,
                'num_frames': self._num_frames,
                'stride': self._video_stride,
                'crop_size': self._frame_size,
                'min_resize': self._min_resize,
                'zero_centering_image': self._zero_centering_image,
                'min_area_ratio': self._min_area_ratio,
                'max_area_ratio': self._max_area_ratio,
                'min_aspect_ratio': self._min_aspect_ratio,
                'max_aspect_ratio': self._max_aspect_ratio,
                'crop_resize_style': self._crop_resize_style,
                'data_path': input_params.data_path,  # Add this line for DMVR
            }

            fps = REF_FPS if ds_name.lower() == 'dmvr' else DEFAULT_FPS
            n_audio_secs = self._num_frames / REF_FPS
            stride = self._video_stride * int(fps // REF_FPS)
            params_factory['stride'] = stride
            self._num_audio_samples = int(REF_SR * n_audio_secs)
            params_factory['num_samples'] = self._num_audio_samples

            if ds_name.lower() == 'dmvr':
                params_factory.update({
                    'output_audio': True,
                    'max_num_words': self._max_num_words,
                    'max_context_sentences': self._max_context_sentences,
                })

            # Get the factory.
            factory_args = {'subset': 'train'}
            factory_class = ds_fctr.get_ds_factory(dataset_name=ds_name)(**factory_args)
            ds_factory = factory_class.configure(**params_factory)

            # Add zeros to audio and/or text if the dataset does not have audio
            # or text already. Also add a boolean to whether audio and/or text
            # are valid and should be used
            ds_factory.postprocessor_builder.add_fn(
                functools.partial(
                    processing.add_audio_text_if_empty,
                    has_valid_text=(ds_name.lower() == 'dmvr'),
                    has_valid_audio=True,
                    num_audio_samples=self._num_audio_samples,
                    max_context_sentences=self._max_context_sentences,
                    max_num_words=self._max_num_words,
                ))

            # Add audio preprocessing.
            if self._audio_noise > 0.:
                ds_factory.preprocessor_builder.add_fn(
                    functools.partial(
                        processing.add_gaussian,
                        gamma=self._audio_noise,
                    ),
                    feature_name=FeatureNames.AUDIO,
                    fn_name='volume_gaussian'
                )

            if self._raw_audio:
                ds_factory.preprocessor_builder.add_fn(
                    processing.extend_waveform_dim,
                    feature_name=FeatureNames.AUDIO,
                    fn_name='extend_waveform',
                )
            else:
                ds_factory.preprocessor_builder.add_fn(
                    functools.partial(
                        processing.raw_audio_to_spectrogram,
                        sample_rate=REF_SR,
                        stft_length=self._stft_length,
                        stft_step=self._stft_step,
                        mel_bins=self._mel_bins,
                        rm_audio=True
                    )
                )
                ds_factory.preprocessor_builder.add_fn(
                    processing.normalize_spectrogram,
                    feature_name=FeatureNames.AUDIO_MEL,
                    fn_name='normalize_mel',
                )

            # Extra data augmentation on video.
            if self._scale_jitter and self._crop_resize_style == 'VGG':
                ds_factory.preprocessor_builder.add_fn(
                    functools.partial(
                        processing.scale_jitter_augm,
                        prob=0.8,
                    ),
                    feature_name=FeatureNames.VISION,
                    fn_name=f'{FeatureNames.VISION}_jitter_scale',
                    add_before_fn_name=f'{FeatureNames.VISION}_resize_smallest'
                )

            ds_factories.append(ds_factory)

        # Add batch-level data-agnostic post-processing functions
        postprocess_fns = []

        if self._space_to_depth:
            postprocess_fns.append(
                functools.partial(
                    processing.space_to_depth,
                    temporal_block_size=2,
                    spatial_block_size=2,
                    feature_name=FeatureNames.VISION,
                )
            )

        if self._linearize_vision:
            postprocess_fns.append(
                functools.partial(
                    processing.linearize,
                    feature_name=FeatureNames.VISION,
                )
            )

        if self._audio_mixup:
            feat_name = FeatureNames.AUDIO if self._raw_audio else FeatureNames.AUDIO_MEL
            postprocess_fns.append(
                functools.partial(
                    processing.batched_mixup,
                    feature_name=feat_name,
                    alpha=self._mixup_alpha,
                    beta=self._mixup_beta,
                    mixup_labels=False,
                )
            )

        num_post_processors = len(postprocess_fns)
        if num_post_processors == 0:
            postprocess_fns = None

        super(PreTrainLoader, self).__init__(
            dmvr_factory=ds_factories,
            params=input_params,
            postprocess_fns=postprocess_fns,
            num_epochs=-1,
            mode='train',
            name=dataset_id,
        )


class EvalLoader(loading.BaseLoader):
    """Constructs the dataloader for evaluation."""

    def __init__(self, dataset_id, params, subset='val', split=None):
        input_params = params.eval.input

        self._num_frames = input_params.num_frames
        self._frame_size = input_params.frame_size
        self._video_stride = input_params.video_stride
        self._raw_audio = input_params.raw_audio
        self._stft_length = input_params.stft_length
        self._stft_step = input_params.stft_step
        self._mel_bins = input_params.mel_bins
        self._zero_centering_image = input_params.zero_centering_image
        self._max_num_words = input_params.max_num_words
        self._max_context_sentences = input_params.max_context_sentences
        self._space_to_depth = input_params.space_to_depth
        self._linearize_vision = input_params.linearize_vision
        self._min_resize = input_params.min_resize

        params_factory = {
            'is_training': False,
            'num_frames': self._num_frames,
            'stride': self._video_stride,
            'crop_size': self._frame_size,
            'min_resize': self._min_resize,
            'zero_centering_image': self._zero_centering_image,
            'data_path': input_params.data_path,  # Add this line for DMVR
        }

        fps = REF_FPS if dataset_id.lower() == 'dmvr' else DEFAULT_FPS
        n_audio_secs = self._num_frames / REF_FPS
        stride = self._video_stride * int(fps // REF_FPS)
        params_factory['stride'] = stride
        self._num_audio_samples = int(REF_SR * n_audio_secs)
        params_factory['num_samples'] = self._num_audio_samples

        if dataset_id.lower() == 'dmvr':
            params_factory.update({
                'output_audio': True,
                'max_num_words': self._max_num_words,
                'max_context_sentences': self._max_context_sentences,
            })

        factory_args = {'subset': subset}
        if split is not None:
            factory_args['split'] = split
        factory_class = ds_fctr.get_ds_factory(dataset_name=dataset_id)(**factory_args)
        ds_factory = factory_class.configure(**params_factory)

        if self._raw_audio:
            ds_factory.preprocessor_builder.add_fn(
                processing.extend_waveform_dim,
                feature_name=FeatureNames.AUDIO,
                fn_name='extend_waveform',
            )
        else:
            ds_factory.preprocessor_builder.add_fn(
                functools.partial(
                    processing.raw_audio_to_spectrogram,
                    sample_rate=REF_SR,
                    stft_length=self._stft_length,
                    stft_step=self._stft_step,
                    mel_bins=self._mel_bins,
                    rm_audio=True
                )
            )
            ds_factory.preprocessor_builder.add_fn(
                processing.normalize_spectrogram,
                feature_name=FeatureNames.AUDIO_MEL,
                fn_name='normalize_mel',
            )

        postprocess_fns = []

        if self._space_to_depth:
            postprocess_fns.append(
                functools.partial(
                    processing.space_to_depth,
                    temporal_block_size=2,
                    spatial_block_size=2,
                    feature_name=FeatureNames.VISION,
                )
            )

        if self._linearize_vision:
            postprocess_fns.append(
                functools.partial(
                    processing.linearize,
                    feature_name=FeatureNames.VISION,
                )
            )

        num_post_processors = len(postprocess_fns)
        if num_post_processors == 0:
            postprocess_fns = None

        super(EvalLoader, self).__init__(
            dmvr_factory=[ds_factory],
            params=input_params,
            postprocess_fns=postprocess_fns,
            num_epochs=1,
            mode='eval',
            name=dataset_id,
        )