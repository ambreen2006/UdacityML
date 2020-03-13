import unittest
from predict import Predict

class PredictTest(unittest.TestCase):

    def test_predict_cpu_with_name_mapping(self):
        probability, index, name = Predict.predict('flowers/sunflower1.jpg',
                                                   'vgg16_flower_classifier.pt',
                                                   'cat_to_name.json',
                                                   False)
        print(name, probability)
        self.assertEqual(name, 'sunflower')

    def test_predict_gpu_with_name_mapping(self):
        probability, index, name = Predict.predict('flowers/sunflower1.jpg',
                                                   'vgg16_flower_classifier.pt',
                                                   'cat_to_name.json',
                                                   True)
        print(name, probability)
        self.assertEqual(name, 'sunflower')

    def test_predict_cpu_with_nonname_mapping(self):
        probability, index, name = Predict.predict('flowers/sunflower1.jpg',
                                                   'vgg16_flower_classifier.pt',
                                                   None,
                                                   False)
        print(name, probability)
        self.assertEqual(name, '')
        self.assertEqual(index, '54')

    def test_predict_gpu_with_nonname_mapping(self):
        probability, index, name = Predict.predict('flowers/sunflower1.jpg',
                                                   'vgg16_flower_classifier.pt',
                                                   None,
                                                   True)
        print(name, probability)
        self.assertEqual(name, '')
        self.assertEqual(index, '54')

        
def suite(self):
    suite = unittest.TestSuite()
    suite.addTest(PredictTest('test_predict_cpu_with_name_mapping'))
    suite.addTest(PredictTest('test_predict_gpu_with_name_mapping'))
    suite.addTest(PredictTest('test_predict_cpu_with_noname_mapping'))
    suite.addTest(PredictTest('test_predict_gpu_with_noname_mapping'))
    return suite
        
def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == "__main__":
    main()
