from comet import download_model
from nltk import download

COMET_MODEL = "wmt20-comet-da"

if __name__ == '__main__':
    download('wordnet')
    download('punkt')
    download('omw-1.4')
    model_path = download_model(COMET_MODEL)
