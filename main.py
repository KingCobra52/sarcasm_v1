import json
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline
from tqdm import tqdm

def get_model_classifications(texts, labels=['not sarcastic', 'sarcastic'], batch_size=20):
    saved_file = Path(
        f'saved_classifications_num-{len(texts)}_labels-'
        f"{'-vs-'.join([s.replace(' ', '-') for s in labels])}"
        '.pkl'
    )
    if not saved_file.exists():
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
        pickle.dump(res, saved_file.open(mode='wb'))
    else:
        print(f'Loading classifications from {str(saved_file)}')
        res = pickle.load(saved_file.open(mode='rb'))
    return res

def main():
    input_strs = []
    targets = []
    for line in open('Sarcasm_Headlines_Dataset_v2.json'):
        d = json.loads(line)
        input_strs.append(d['headline'])
        targets.append(d['is_sarcastic'])
    num_samples = len(input_strs)
    possible_labels = [
        ['not sarcastic', 'sarcastic'],
        ['genuine', 'not serious'],
        ['real', 'fake']
    ]
    accuracies = []
    for real_label_name, fake_label_name in possible_labels:
        results = get_model_classifications(input_strs[: num_samples])
        predictions = np.array(results) == fake_label_name
        acc = (predictions == targets[: num_samples]).mean() * 100
        print(f'Accuracy: {acc:.3f}%')
        accuracies.append(acc)
    plt.bar(range(len(accuracies)), accuracies)
    plt.show()

if __name__ == '__main__':
    main()
