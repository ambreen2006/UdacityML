from argparse import ArgumentParser
from flower_classifier import FlowerClassifier

class Predict:

    @staticmethod
    def restore_model(check_point_path, category_names_path, use_gpu):
        flowerc = FlowerClassifier(check_point_path, category_names_path, use_gpu)
        return flowerc

    def topk(image_path, check_point_path, category_names_path, topk, use_gpu):
        flowerc = Predict.restore_model(check_point_path, category_names_path)
        flowerc.topk(image_path, topk, use_gpu)

    def predict(image_path, check_point_path, category_names_path, use_gpu):
        flowerc = Predict.restore_model(check_point_path, category_names_path, use_gpu)
        return flowerc.predict(image_path)

def main():

    parser = ArgumentParser(description='Predict type of flower given the image and the checkpoint for the saved model')
    parser.add_argument('image_path', help = 'path of the image to be classified')
    parser.add_argument('check_point', help = 'path of the trained model')
    parser.add_argument('--gpu', action = 'store_true', default = False , help = 'use gpu')
    parser.add_argument('--topk', action = 'store', type=int, help = 'get top k predictions')
    parser.add_argument('--category_names', action='store', help = 'provide path to json containing category to name mapping')
    
    args = parser.parse_args()
    
    if args.topk:
        self.topk(args.image_path, args.check_point, args.category_names,  args.topk, args.gpu)
    else:
        prob, index, name = Predict.predict(args.image_path, args.check_point, args.category_names, args.gpu)
        print("Probability = {}, Index = {}, Name = {}".format( prob, index, name))

if __name__ == "__main__":
    main()

