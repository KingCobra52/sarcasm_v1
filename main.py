import json

from transformers import pipeline
from tqdm import tqdm

def get_model_classifications(texts, labels=['not sarcastic', 'sarcastic'], batch_size=20):
    classifier = pipeline(
        "zero-shot-classification",
        model="sileod/deberta-v3-base-tasksource-nli"
    )
    res = []
    it = tqdm(
        range(len(texts) // batch_size + 1),
        desc=f'Processing batches of size {batch_size}'
    )
    for batch_idx in it:
        text_batch = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        clses = classifier(text_batch, labels)
        res += [
            d['labels'][
                max(enumerate(d['scores']), key=lambda ix: (ix[1], ix[0]))[0]
            ]
            for d in clses
        ]
    return res

def main():
    input_strs = []
    targets = []
    for line in open('Sarcasm_Headlines_Dataset_v2.json'):
        d = json.loads(line)
        input_strs.append(d['headline'])
        targets.append(d['is_sarcastic'])
    results = get_model_classifications(input_strs)
    print(f'Model guesses: {results}')
    print(f'True answers: {targets}')

if __name__ == '__main__':
    main()
