import sys
import json
import jsonlines
import os
from glob import glob
from datasets import load_metric
from comet import download_model, load_from_checkpoint
from nltk import word_tokenize
import scispacy
import spacy
import logging

BLEU = 'bleu'
ROUGE = 'rouge'
SACREBLEU = 'sacrebleu'
METEOR = 'meteor'
COMET = 'comet'
NER_ACCURACY = 'ner_accuracy'
COMET_MODEL = "wmt21-cometinho-da"


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
    nlp = spacy.load("en_core_sci_sm", exclude=["ner", "parser"])
    pos_tags = ['PROPN', 'VERB', 'NOUN', 'ADJ']
    for ref, pred in zip(target_ref, target_pred):
        accuracy = 1.0
        ner_reference = data.get(ref, [])
        lemmas = ' '.join([word.lemma_ for word in nlp(pred) if
                           word.pos_ in pos_tags])
        if ner_reference:
            accuracy = 0
            for entity in ner_reference:
                entity_text = ' '.join(
                    [word.lemma_ for word in nlp(entity['text']) if
                     word.pos_ in pos_tags])
                if (entity_text.lower() in lemmas.lower()) or (
                        entity['text'] in pred):
                    accuracy += 1
                else:
                    logging.info(
                        f"named entity not translated: {entity['text']},"
                        f" {pred}, {entity['labels'][0]}")
            accuracies.append(accuracy / len(ner_reference))
        else:
            accuracies.append(accuracy)
    return accuracies, sum(accuracies) / len(target_pred)


def write_outfile(results, output_dir, source_language, target_language):
    logging.info(results)
    output_filename = os.path.join(output_dir, 'scores.txt')
    out_strings = [
        # f"METEOR:{results[METEOR][METEOR]}\n",
        f"COMET:{results[COMET][-1]}\n",
        f"BLEU:{results[BLEU][BLEU]}\n",
        # f"SACREBLEU:{results[SACREBLEU]['score']}\n",
        f"NER-accuracy:{results[NER_ACCURACY][-1]}\n",
        f"ROUGE-L-F1:{results[ROUGE]['rougeL'].mid.fmeasure}\n",
    ]
    with open(output_filename, 'wb') as f:
        f.write(''.join(out_strings).encode('utf8'))


def main():
    input_dir, output_dir = sys.argv[1], sys.argv[2]
    submit_dir = os.path.join(input_dir, 'res')
    truth_dir = os.path.join(input_dir, 'ref')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    answer_path = glob(f"{submit_dir}/*.json*")[0]
    source_language, target_language = "ru", "en"
    truth_file = os.path.join(truth_dir, "references.jsonl")
    ner = os.path.join(truth_dir, 'ner.json')
    target_ref, source_ref = read_jsonlines(truth_file,
                                            source_language,
                                            target_language)
    target_pred, source_pred = read_jsonlines(answer_path,
                                              source_language,
                                              target_language)
    assert source_pred == source_ref
    assert len(target_ref) == len(target_pred)
    bleu_references = [word_tokenize(text) for text in target_ref]
    bleu_references = [[ref] for ref in
                       bleu_references]  # allows many true translations
    bleu_predictions = [word_tokenize(text) for text in target_pred]
    sacrebleu_references = [[ref] for ref in target_ref]
    model_path = download_model(COMET_MODEL)
    model = load_from_checkpoint(model_path)
    logging.info("Comet checkpoint loaded")
    comet_data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in
                  zip(source_ref, target_pred, target_ref)]
    results = {BLEU: compute_metric(BLEU, bleu_references, bleu_predictions),
               ROUGE: compute_metric(ROUGE, target_ref, target_pred),
               SACREBLEU: compute_metric(SACREBLEU, sacrebleu_references,
                                         target_pred),
               METEOR: compute_metric(METEOR, target_ref, target_pred),
               NER_ACCURACY: ner_accuracy(ner, target_ref, target_pred),
               COMET: model.predict(comet_data, gpus=0),
               }
    write_outfile(results, output_dir, source_language, target_language)


if __name__ == '__main__':
    main()
