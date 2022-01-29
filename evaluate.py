import argparse
import json
import jsonlines
from datasets import load_metric
import sentencepiece as spm
from comet import download_model, load_from_checkpoint
from nltk import word_tokenize

BLEU = 'bleu'
ROUGE = 'rouge'
SACREBLEU = 'sacrebleu'
METEOR = 'meteor'
COMET = 'comet'
NER_ACCURACY = 'ner_accuracy'
COMET_MODEL = "wmt20-comet-da"


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation script')
    parser.add_argument('references', type=str)
    parser.add_argument('predictions', type=str)
    parser.add_argument('ner', type=str)
    parser.add_argument('source_language', choices=['en', 'ru'])
    parser.add_argument('target_language', choices=['en', 'ru'])
    return parser.parse_args()


def read_jsonlines(path, source_language, target_language):
    with jsonlines.open(path, 'r') as reader:
        target, source = [], []
        for obj in reader:
            target.append(obj['translation'][target_language])
            source.append(obj['translation'][source_language])
        return target, source


def compute_metric(metric_name, references, predictions):
    return load_metric(metric_name).compute(predictions=predictions,
                                            references=references)


def ner_accuracy(ner, target_ref, target_pred):
    with open(ner, 'r', encoding='utf8') as f:
        data = json.load(f)
    accuracies = []
    for ref, pred in zip(target_ref, target_pred):
        accuracy = None
        ner_reference = data.get(ref, [])
        tokenized = word_tokenize(pred)
        if ner_reference:
            accuracy = 0
            for entity in ner_reference:
                if entity['text'] in tokenized:
                    accuracy += 1
            accuracies.append(accuracy/len(ner_reference))
        else:
            accuracies.append(accuracy)
    return accuracies, \
        sum([acc for acc in accuracies if acc is not None])/len(target_pred)


def main():
    args = parse_args()
    target_ref, source_ref = read_jsonlines(args.references,
                                            args.source_language,
                                            args.target_language)
    target_pred, source_pred = read_jsonlines(args.predictions,
                                              args.source_language,
                                              args.target_language)
    assert source_pred == source_ref
    assert len(target_ref) == len(target_pred)
    sp = spm.SentencePieceProcessor(model_file=f'{args.target_language}.spm')
    bleu_references = sp.encode(target_ref, out_type=str)
    bleu_references = [[ref] for ref in
                       bleu_references]  # allows many true translations
    bleu_predictions = sp.encode(target_pred, out_type=str)
    sacrebleu_references = [[ref] for ref in target_ref]
    model_path = download_model(COMET_MODEL)
    model = load_from_checkpoint(model_path)
    comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in
                  zip(source_ref, target_pred, target_ref)]
    results = {BLEU: compute_metric(BLEU, bleu_references, bleu_predictions),
               ROUGE: compute_metric(ROUGE, target_ref, target_pred),
               SACREBLEU: compute_metric(SACREBLEU, sacrebleu_references,
                                         target_pred),
               METEOR: compute_metric(METEOR, target_ref, target_pred),
               NER_ACCURACY: ner_accuracy(args.ner, target_ref, target_pred),
               COMET: model.predict(comet_data, gpus=0),
               }
    with open('results.json', 'w', encoding='utf8') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
