from eywa.nn import NNClassifier
import pytest

def test_nn_classifier_basic():
    
    docs=['book for a holiday place','any nice places to spend the night','rent a hotel room','find a place to stay','what is the weather in kochi','weather america','will it snow today','it is a sunny day']
    labels=['hotel','hotel','hotel','hotel','weather','weather','weather','weather']
    nnclf=NNClassifier(docs, labels)
    x_tests = ['book for a place to stay',
                'rent a holiday place','weather germany','will it rain today']
    y_tests = ['hotel','hotel','weather','weather']
    for x,y in zip(x_tests, y_tests):
        assert nnclf.predict(x) == y
if __name__ == '__main__':
    pytest.main([__file__])