from data_gen.tts.base_binarizer import BaseBinarizer
import re
from copy import deepcopy
import logging
from data_gen.tts.binarizer_zh import ZhBinarizer
from utils.hparams import hparams


def split_train_test_set(item_names):
    item_names = deepcopy(item_names)
    test_item_names = [x for x in item_names if any([ts in x for ts in hparams['test_prefixes']])]
    train_item_names = [x for x in item_names if x not in set(test_item_names)]
    logging.info("train {}".format(len(train_item_names)))
    logging.info("test {}".format(len(test_item_names)))
    return train_item_names, test_item_names


class SingingBinarizer(ZhBinarizer, BaseBinarizer):
    def __init__(self):
        super(SingingBinarizer, self).__init__()
        new_item_names = []
        n_utt_ds = {k: 0 for k in hparams['datasets']}
        for item_name in self.item_names:
            for dataset in hparams['datasets']:
                if len(re.findall(rf'{dataset}', item_name)) > 0:
                    new_item_names.append(item_name)
                    n_utt_ds[dataset] += 1
                    break
        print(n_utt_ds)
        self.item_names = new_item_names
        self._train_item_names, self._test_item_names = split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._test_item_names


if __name__ == "__main__":
    SingingBinarizer().process()
