import importlib
import os
import json
import torch
from ts.torch_handler.image_classifier import ImageClassifier
from ts.utils.util import map_class_to_label
from ts.utils.util import list_classes_from_module


class EfficientNetImageClassifier(ImageClassifier):
    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        setup_config_path = os.path.join(model_dir, 'setup_config.json')
        if not os.path.isfile(setup_config_path):
            raise RuntimeError("Missing the setup_config.json file")

        with open(setup_config_path) as file:
            setup_config = json.load(file)

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError("Expected only one class as model definition. {}".format(
                model_class_definitions))

        model_class = model_class_definitions[0]
        self.num_classes = setup_config.get('num_classes', 1000)
        model = model_class.from_pretrained(setup_config['model_name'],
                                            weights_path=model_pt_path,
                                            num_classes=self.num_classes)
        return model

    def postprocess(self, data):
        if self.num_classes > 1:
            return super().postprocess(data)
        prob = torch.sigmoid(data)
        probs = torch.cat([1-prob, prob], dim=-1).tolist()
        return map_class_to_label(probs, self.mapping)
