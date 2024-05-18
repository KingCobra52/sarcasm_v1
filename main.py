import json

def main():
    v1 = set()
    for line in open('Sarcasm_Headlines_Dataset.json'):
        d = json.loads(line)
        v1.add(d['headline'])

    v2 = set()
    for line in open('Sarcasm_Headlines_Dataset_v2.json'):
        d = json.loads(line)
        v2.add(d['headline'])

    print(v2 - v1)

if __name__ == '__main__':
    main()
