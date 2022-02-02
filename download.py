import logging
from datasets import load_metric
from comet import download_model, load_from_checkpoint
from nltk import download

COMET_MODEL = "wmt21-cometinho-da"

if __name__ == '__main__':
    download('wordnet')
    download('punkt')
    download('omw-1.4')
    model_path = download_model(COMET_MODEL)
    logging.info(model_path)
    load_from_checkpoint(model_path)  # in order to run all from_pretrained
    for metric in ('bleu', 'rouge', 'sacrebleu', 'meteor'):
        load_metric(metric)  # even this downloads something
