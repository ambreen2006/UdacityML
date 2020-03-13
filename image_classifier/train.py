from argparse import ArgumentParser
from flower_classifier import ImageClassiferForTraining

class Train:

    @staticmethod
    def create_classifier(data_directory, attributes):
        try:
            classifier = ImageClassiferForTraining(data_directory, attributes)
            return classifier
        except Exception as e: raise

    @staticmethod
    def train(data_directory, epochs, attributes):
        classifier = Train.create_classifier(data_directory, attributes)
        classifier.train(epochs)

def main():
    
    parser = ArgumentParser(description = 'Train classifier')
    helpful_text = {
        "data_directory":"Path to the training data",
        "save_dir":"Path to store the checkpoint",
        "arch":"Pretrained network architecture",
        "learning_rate":"Learning rate for the network",
        "epochs":"# of epochs",
        "hidden_units":"# of hidden units",
        "gpu":"Use GPU"
    }

    parser.add_argument('data_directory', help = helpful_text["data_directory"])
    parser.add_argument('--save_dir', action = 'store', help  = helpful_text["save_dir"])
    parser.add_argument('--arch', help = helpful_text["arch"])
    parser.add_argument('--learning_rate', action = 'store', type = float, help = helpful_text["learning_rate"])
    parser.add_argument('--hidden_units', action = 'store', type = int, nargs='+', help = helpful_text["hidden_units"])
    parser.add_argument('--epochs', action = 'store', type = int, help = helpful_text["epochs"])
    parser.add_argument('--gpu', action = 'store_true', default = False , help = helpful_text["gpu"])

    attributes = {}

    args = parser.parse_args()
    data_directory = args.data_directory

    epochs = 1

    if args.arch:
        attributes['arch'] = args.arch
    if args.learning_rate:
        attributes['learning_rate'] = args.learning_rate
    if args.save_dir:
        attributes['save_dir'] = args.save_dir
    if args.gpu:
        attributes['use_gpu'] = True
    if args.epochs:
        epochs = args.epochs
    if args.hidden_units:
        attributes['hidden_layers'] = args.hidden_units

    Train.train(data_directory, epochs, attributes)
    
if __name__ == "__main__":
    main()
