from .classifier import Classifier
from ..nn import NNPicker
from ..lang import Document, Token

class EntityExtractor(object):

    def __init__(self):
        self.X = []
        self.Y = []
        self._changed = False
        self.clfs = {}
        self.picker = None
        self._training_config = None

    def fit(self, X, Y, training_config=None):
        x_app = self.X.append
        y_app = self.Y.append
        if not isinstance(X, (list, tuple)):
            X = [X]
        if not isinstance(Y, (list, tuple)):
            Y = [Y]
        assert len(X) == len(Y)
        for x, y in zip(X, Y):
            x = Document(x)
            assert isinstance(y, dict)
            for k, v in y.items():
                assert isinstance(k, str)
                assert isinstance(v, (str, type(None)))
            x_app(x)
            y_app(y)
        self._changed = True
        self._training_config = training_config

    def _compile(self):
        self.keys = set()
        for x, y in zip(self.X, self.Y):
            for k, v in y.items():
                self.keys.add(k)
                clf = self.clfs.get(k)
                if clf is None:
                    clf = Classifier()
                    self.clfs[k] = clf
                if v and v in x:
                    clf.fit(x, None)
                else:
                    clf.fit(x, v)
        self.picker = NNPicker(self.X, self.Y, training_config=self._training_config)

    def predict(self, x, keys=None, return_scores=False):
        if self._changed:
            self._compile()
            self._changed = False
        if type(x) in (list, tuple):
            return type(x)(map(lambda x:
                               self.predict(x, keys, return_scores), x))
        if keys is None:
            keys = list(self.keys)
        keys_to_pick = []
        values = {}
        for k in keys:
            v = self.clfs[k].predict(x)
            if v:
                values[k] = v
            else:
                keys_to_pick.append(k)
        values.update(self.picker.predict(x, keys=keys_to_pick, multiple=False))
        return values


    def serialize(self):
        """Serializes the `EntityExtractor` object to a json
        friendly config.

        # Returns
        `dict`
        """
        if self._changed:
            self._compile()
            self._changed = False
        else:
            if hasattr(self, '_config'):
                return self._config
        config = {}
        config['class'] = self.__class__.__name__
        config['X'] = [str(x) for x in self.X]
        config['Y'] = self.Y[:]
        config['clfs'] = {k : v.serialize() for k, v in self.clfs.items()}
        config['picker'] = self.picker.serialize()
        self._config = config
        return config

    @classmethod
    def deserialize(cls, config):
        """Deserializes a `EntityExtractor` config to a `EntityExtractor` instance.

        # Arguments
        config: `dict`. `EntityExtractor` config (generated by `EntityExtractor.serialize`).

        # Returns
        `EntityExtractor` instance
        """
        ee = cls()
        ee.X = list(map(Document, config['X']))
        ee.Y = config['Y']
        ee.keys = set()
        for y in ee.Y:
            for k in y:
                ee.keys.add(k)
        ee.clfs = {k : Classifier.deserialize(v) for k, v in config['clfs'].items()}
        ee.picker = NNPicker.deserialize(config['picker'])
        ee._changed = False
        return ee