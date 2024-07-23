# vatt/data/datasets/dmvr_dataset.py

import functools
import tensorflow as tf
from dmvr import builders
from dmvr import processors
from vatt.data import processing

FeatureNames = processing.FeatureNames

class DMVRFactory:
    def __init__(self, subset='train'):
        self._subset = subset
        self._num_classes = None  # Set this if you have a fixed number of classes

    def configure(self, **config):
        self._config = config
        return self

    def make_dataset(self):
        data_path = self._config.get('data_path', r'C:\dmvr\examples\dataset_dmvr')
        if not data_path:
            raise ValueError("data_path must be provided in the configuration")

        dataset = tf.data.TFRecordDataset(data_path)
        
        parser_builder = builders.SequenceExampleParserBuilder()
        
        # Define your features here. This is an example and should be adjusted
        # to match your DMVR record structure
        parser_builder.parse_feature(
            feature_name=FeatureNames.VISION,
            feature_type=tf.io.FixedLenSequenceFeature((), tf.string),
            output_dtype=tf.string)
        parser_builder.parse_feature(
            feature_name=FeatureNames.AUDIO,
            feature_type=tf.io.FixedLenSequenceFeature((), tf.float32),
            output_dtype=tf.float32)
        parser_builder.parse_feature(
            feature_name=FeatureNames.TEXT_INDEX,
            feature_type=tf.io.FixedLenSequenceFeature((), tf.int64),
            output_dtype=tf.int32)
        
        # Add more features as needed
        
        processor_builder = builders.ProcessorBuilder(parser_builder)

        # Video processing
        processor_builder.add_fn(
            fn=processors.decode_jpeg,
            feature_name=FeatureNames.VISION,
            fn_name=f'{FeatureNames.VISION}_decode',
            )
        processor_builder.add_fn(
            fn=functools.partial(
                processors.sample_sequence,
                num_steps=self._config.get('num_frames', 32),
                stride=self._config.get('video_stride', 1),
                seed=None),
            feature_name=FeatureNames.VISION,
            fn_name=f'{FeatureNames.VISION}_sample',
            )
        processor_builder.add_fn(
            fn=functools.partial(
                processors.resize_smallest,
                size=self._config.get('frame_size', 224)),
            feature_name=FeatureNames.VISION,
            fn_name=f'{FeatureNames.VISION}_resize_smallest',
            )

        # Audio processing
        if self._config.get('raw_audio', True):
            processor_builder.add_fn(
                fn=functools.partial(
                    processors.sample_sequence,
                    num_steps=self._config.get('num_audio_samples', 48000),
                    stride=self._config.get('audio_stride', 1),
                    seed=None),
                feature_name=FeatureNames.AUDIO,
                fn_name=f'{FeatureNames.AUDIO}_sample',
                )
        else:
            # Add spectrogram processing if needed
            pass

        # Text processing
        processor_builder.add_fn(
            fn=functools.partial(
                processors.sample_sequence,
                num_steps=self._config.get('max_num_words', 16),
                stride=1,
                seed=None),
            feature_name=FeatureNames.TEXT_INDEX,
            fn_name=f'{FeatureNames.TEXT_INDEX}_sample',
            )

        # Add more processing steps as needed

        postprocessor_builder = builders.PostprocessorBuilder(processor_builder)

        # Add any postprocessing steps here
        # For example, you might want to add data augmentation for training

        if self._subset == 'train':
            postprocessor_builder.add_fn(
                fn=functools.partial(
                    processors.random_crop_resize,
                    output_size=self._config.get('frame_size', 224),
                    num_views=1,
                    seed=None),
                feature_name=FeatureNames.VISION,
                fn_name=f'{FeatureNames.VISION}_random_crop_resize')
        else:
            postprocessor_builder.add_fn(
                fn=functools.partial(processors.central_crop_resize,
                                     output_size=self._config.get('frame_size', 224)),
                feature_name=FeatureNames.VISION,
                fn_name=f'{FeatureNames.VISION}_central_crop')

        # Build the dataset
        dataset = postprocessor_builder.make_dataset(dataset)

        # Batching
        batch_size = self._config.get('batch_size', 32)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # Prefetching
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    @property
    def num_classes(self):
        return self._num_classes