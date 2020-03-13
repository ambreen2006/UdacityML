import unittest
from train import Train

class PredictTest(unittest.TestCase):

    def test_train_default_arch(self):
        classifier = Train.create_classifier('flowers', 'vgg16')

    def test_train_unsupported_arch(self):
        with self.assertRaises(Exception) as e:
            Train.create_classifier('flowers', 'whatever')

    def test_train(self):
        classifier = Train.create_classifier('flowers', 'vgg16')
        classifier.train(1)

        
def suite(self):
    suite = unittest.TestSuite()
    suite.addTest(PredictTest('test_train_default_arch'))
    suite.addTest(PredictTest('test_train_unsupported_arch'))
    suite.addTest(PredictTest('test_train'))
    #suite.addTest(PredictTest('test_predict_cpu_with_noname_mapping'))
    #suite.addTest(PredictTest('test_predict_gpu_with_noname_mapping'))
    return suite
        
def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())

if __name__ == "__main__":
    main()
