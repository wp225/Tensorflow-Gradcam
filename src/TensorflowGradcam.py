from typing import List

import PIL.Image as Image
import numpy as np
import tensorflow
import tensorflow.keras as keras
from matplotlib.pyplot import get_cmap as cm
from pydantic import BaseModel, ConfigDict
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array


class GradcamInput(BaseModel):
    model: Model
    image: Image.Image
    normalized: bool

    class Config(ConfigDict):
        arbitrary_types_allowed = True


class TensorflowGradcam:
    def __init__(self,inputs: GradcamInput):
        self.model = inputs.model
        self.image = inputs.image
        self.normalized = inputs.normalized
        self.image = self.image.convert('RGB')


    def preprocess_image(self):

        input_shape = model.input.shape[1:3]
        image = self.image.resize(size=input_shape)
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)

        if not self.normalized:
            return image_array

        image_array= image_array/225.0
        return image_array



    def get_last_layer(self) -> tuple[keras.layers.Conv2D, List[keras.layers.Layer]]:
        """

        :return: Last Conv Layer and a List of Layers that follow it
        """
        last_conv_layer = None
        classifier_layers: List[Model] = []
        for layer in reversed(self.model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv_layer = layer
                break
            else:
                classifier_layers.append(layer)

        classifier_layers.reverse()

        return last_conv_layer, classifier_layers

    def gradcam(self) -> tuple[str, Image]:
        """

        :return: Prediction class for given image, PIL image
        """
        image = self.preprocess_image()
        last_conv_layer, classifier_layers = self.get_last_layer()
        if last_conv_layer is None:
            return 'No Conv Layer found to get gradients from'

        last_conv_layer_model = Model(
            inputs=self.model.inputs,
            outputs=last_conv_layer.output
        )

        print(last_conv_layer_model.summary())

        last_conv_layer_model_output = keras.Input(last_conv_layer_model.output.shape[1:])
        x = last_conv_layer_model_output
        for layers in classifier_layers:
            x = layers(x)

        classifier_model = Model(inputs=last_conv_layer_model_output, outputs=x)
        print(classifier_model.summary())

        with tensorflow.GradientTape() as tape:

            last_conv_layer_model_output = last_conv_layer_model(image)
            tape.watch(last_conv_layer_model_output)  # Watch operation last_conv_layer_model_output

            # Prediction for each image(batch) for all classes. Eg: [[0.1 0.2 0.3]] for single image w 3 classes
            predictions = classifier_model(last_conv_layer_model_output)
            # Index for max prediction score. For single image use 0. Eg: 3 for above example
            top_prediction_index = np.argmax(predictions[0])
            # Score of max index. Eg: 0.3
            top_class_channel = predictions[:, top_prediction_index]

        grads = tape.gradient(top_class_channel, last_conv_layer_model_output)
        # reduced gradients across batch(0),height(1),width(2) channel ->(512,) dims
        pooled_grad = tensorflow.reduce_mean(grads, axis=[0, 1, 2])

        last_conv_layer_model_output = last_conv_layer_model_output.numpy()[0]
        pooled_grad = pooled_grad.numpy()

        for i in range(pooled_grad.shape[-1]):
            last_conv_layer_model_output[:, :, i] *= pooled_grad[i]

        print(last_conv_layer_model_output.shape)
        heatmap = np.mean(last_conv_layer_model_output, axis=-1)
        normalized_heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        normalized_heatmap = np.uint8(255 * normalized_heatmap)

        jet = cm('jet')
        jet_color = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_color[normalized_heatmap]

        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((self.image.width,self.image.height))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap * 0.4 + self.image

        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        return top_prediction_index,superimposed_img


if __name__ == '__main__':
    image = Image.open('../tiger.jpeg')
    model = keras.applications.vgg16.VGG16(weights='imagenet')
    normalized = False
    inputs = GradcamInput(model=model, image=image, normalized=normalized)
    test = TensorflowGradcam(inputs)
    b,a,img = test.gradcam()
    print(b,a)
    img.save('./test.png')
