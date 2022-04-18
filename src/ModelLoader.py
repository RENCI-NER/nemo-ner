import os
from nemo.collections.nlp.models import TokenClassificationModel


def load_model():
    path = os.environ.get('MODEL_PATH', os.path.join(os.path.dirname(__file__), '..', 'trained_model.nemo'))
    model = TokenClassificationModel.restore_from(path)
    predictions = model.add_predictions(
        ["ashtma is a disease"]
    )
    print(predictions)


if __name__ == '__main__':
    load_model()