import os
import json
import random
import datasets
import pandas as pd

from transformers import BertForMaskedLM, BertConfig, BertTokenizer


TASK_CLASSES = {
    'agnews': {
        'metric_prompt': [lambda text1, text2: [text1] + ['[SEP]', 'A', 'news', 'of', '[MASK]', 'topic', ':'] + [text2]],
    },
    'yahoo_answers_topics': {
        'metric_prompt': [lambda text1, text2: [text1] + ['[SEP]', 'A', 'news', 'of', '[MASK]', 'topic', ':'] + [text2]],
    },
    'dbpedia': {
        'metric_prompt': [lambda text1, text2: [text1] + ['[SEP]', 'A', 'news', 'of', '[MASK]', 'topic', ':'] + [text2]],
    },
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'mlm': BertForMaskedLM
    },
}


def load_true_few_shot_dataset(args):

    labels = []
    with open(os.path.join(args.data_path, 'TextClassification', args.dataset, 'classes.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                labels.append(line)

    if args.dataset == 'agnews':
        test_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'test.csv'), header=None)
        test_data = []
        for i in range(1, test_df.shape[0]):
            line = test_df.loc[i]
            index, text, label_id = list(line)
            label_id = int(label_id)
            # label_id -= 1
            # raw = title + '. ' + text
            # text = text[:200]
            text = ' '.join(text.split()[:120])
            item = {
                # 'title': title,
                # 'text': text,
                'raw': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            test_data.append(item)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            path = os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'train.csv'), header=None)
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(1, train_df.shape[0]):
            line = train_df.loc[i]
            index, text, label_id = list(line)
            label_id = int(label_id)
            # label_id -= 1
            # raw = title + '. ' + text
            # text = text[:200]
            text = ' '.join(text.split()[:120])
            item = {
                # 'title': title,
                # 'text': text,
                'raw': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            train_data_dict[label_id].append(item)

    elif args.dataset == 'yahoo_answers_topics':
        test_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'test.csv'), header=None)
        test_data = []
        for i in range(test_df.shape[0]):
            line = test_df.loc[i]
            label_id, question_title, question_content, answer = list(line)
            raw = question_title + question_content + answer
            raw = ' '.join(raw.split()[:120])
            label_id = int(label_id)
            item = {
                'raw': raw,
                'label': str(label_id),
                'text_len': len(raw)
            }
            test_data.append(item)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'),
                      'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'train.csv'), header=None)
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(train_df.shape[0]):
            line = train_df.loc[i]
            label_id, question_title, question_content, answer = list(line)
            raw = question_title + question_content + answer
            raw = ' '.join(raw.split()[:120])
            label_id = int(label_id)
            item = {
                'raw': raw,
                'label': str(label_id),
                'text_len': len(raw)
            }
            train_data_dict[label_id].append(item)

    else:
        test_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'test.csv'), header=None)
        test_data = []
        for i in range(test_df.shape[0]):
            line = test_df.loc[i]
            _, label_id, title, content = list(line)
            raw = title + content
            raw = ' '.join(raw.split()[:120])
            try:
                label_id = int(label_id)
            except:
                continue
            item = {
                'raw': raw,
                'label': str(label_id),
                'text_len': len(raw)
            }
            test_data.append(item)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'),
                      'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'train.csv'), header=None)
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(train_df.shape[0]):
            line = train_df.loc[i]
            _, label_id, title, content = list(line)
            raw = title + content
            raw = ' '.join(raw.split()[:120])
            try:
                label_id = int(label_id)
            except:
                continue
            item = {
                'raw': raw,
                'label': str(label_id),
                'text_len': len(raw)
            }
            train_data_dict[label_id].append(item)

    episodes = []

    for _ in range(10):
        train_data = []
        for i in range(len(labels)):
            this_samples = random.sample(train_data_dict[i], args.k_shot)
            train_data = train_data + this_samples
        
        # replace some training samples’ labels randomly to introduce noises
        wrong_num = args.wrong_num
        wrong_indexes = random.sample(range(len(train_data)), wrong_num)
        for index in wrong_indexes:
            train_data[index]['label'] = str(random.sample(range(len(labels)), 1))

        episode_to_save = {
            'support_set': train_data,
        }
        with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'), 'a') as f:
            string = json.dumps(episode_to_save)
            f.write(string)
            f.write('\n')

        episode = {
            'support_set': train_data,
            'query_set': test_data,
            'labels': labels
        }
        episodes.append(episode)


    return episodes
