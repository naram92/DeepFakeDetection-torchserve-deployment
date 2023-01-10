from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import load_pretrained_weights


class ImageClassifier(EfficientNet):
    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """Create an efficientnet model according to the given name
        Args:
            model_name (str): The name of the efficientnet model to create
            weights_path (str or None):
                str: The path to the pretrained weights file on the local disk
                None: Use pretrained weights downloaded from the internet
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=True, advprop=advprop)
        model._change_in_channels(in_channels)
        model.eval()
        return model
