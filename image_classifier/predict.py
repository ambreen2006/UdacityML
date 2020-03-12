from argparse import ArgumentParser
from flower_classifier import FlowerClassifier

def restore_model(check_point):
    flowerc = FlowerClassifier()
    flowerc.restore(check_point)
    return flowerc

def topk(image_path, check_point, topk, use_gpu):
    flowerc = restore_model(check_point)
    flowerc.topk(image_path, topk, use_gpu)

def predict(image_path, check_point, use_gpu):
    flowerc = restore_model(check_point)
    return flowerc.predict(image_path, use_gpu)

parser = ArgumentParser(description='Predict type of flower given the image and the checkpoint for the saved model')
parser.add_argument('image_path', help = 'path of the image to be classified')
parser.add_argument('check_point', help = 'path of the trained model')
parser.add_argument('--gpu', action = 'store_true', default = False , help = 'use gpu')
parser.add_argument('--topk', action = 'store', type=int, help = 'get top k predictions')

args = parser.parse_args()

print("topk", args.topk)

if args.topk:
    topk(args.image_path, args.check_point, args.topk, args.gpu)
else:
    prob, index, name = predict(args.image_path, args.check_point, args.gpu)
    print(name, prob)
