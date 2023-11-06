import fire
import json
import numpy as np


def merge(file1_path, file2_path, output_path='merged.json'):
    content = []
    with open(file1_path, 'r') as f:
        for line in f:
            content.append(json.loads(line))

    with open(file2_path, 'r') as f:
        for line in f:
            content.append(json.loads(line))

    np.random.shuffle(content)

    with open(output_path, 'w') as f:
        for line in content:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    fire.Fire(merge)
