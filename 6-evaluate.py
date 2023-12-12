"""This module is responsible for evaluating the model and storing the results in the `out/eval` folder."""
import numpy as np
from src import model_builder, mk_dataset_df
import warnings

warnings.filterwarnings("ignore")
def process_image(image):
    """Process the image."""
    image = image.resize((224, 224))
    # convert to rgb
    image = image.convert("RGB")
    return np.array(image)
if __name__ == "__main__":
    df = mk_dataset_df("data/raw/Fruits Classification/train")
    X = df[["image"]]
    img_size = (224, 224)
    X = np.array([process_image(x) for x in X["image"]])
    print(X.shape)
    model = model_builder(X[0].shape, 5)

    y = df["label"]
    # cross validation
    res = model.evaluate(X, y)
    print({metric: value for metric, value in zip(model.metrics_names, res)})
