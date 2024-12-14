import sys
import os
import pickle
import random
import json
import torch
import torch.utils.data as data
import torch.nn as nn

sys.path.append('music-gen')
import third_party.midi_processor.processor as midi_processor

SEQUENCE_START = 0
RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100
TOKEN_END = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD = TOKEN_END + 1
VOCAB_SIZE = TOKEN_PAD + 1

def process_midi(raw_mid, max_seq, random_seq):
    x  = torch.full((max_seq, ), TOKEN_PAD, dtype = torch.long)
    tgt = torch.full((max_seq, ), TOKEN_PAD, dtype = torch.long)

    raw_len = len(raw_mid)
    full_seq = max_seq + 1 

    if(raw_len == 0):
        return x, tgt

    if(raw_len < full_seq):
        x[:raw_len] = raw_mid
        tgt[:raw_len-1] = raw_mid[1:]
        tgt[raw_len-1]  = TOKEN_END
    else:
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(SEQUENCE_START, end_range)
        else:
            start = SEQUENCE_START

        end = start + full_seq
        data = raw_mid[start:end]
        x = data[:max_seq]
        tgt = data[1:full_seq]
    return x, tgt

def process_jsb(raw_jsb, max_seq, random_seq = True):
    x = torch.full((max_seq, ), 90, dtype = torch.long)
    tgt = torch.full((max_seq, ), 90, dtype = torch.long)
    raw_len = len(raw_jsb)
    full_seq = max_seq + 1

    if(raw_len < full_seq):
        x[:raw_len] = raw_jsb
        tgt[:raw_len-1] = raw_jsb[1:]
        tgt[raw_len-1]  = 91
    else:
        if(random_seq):
            end_range = raw_len - full_seq
            start = random.randint(0, end_range)
        else:
            start = 0
        end = start + full_seq
        data = raw_jsb[start:end]
        x = data[:max_seq]
        tgt = data[1:full_seq]
    x[x == -1] = 89
    tgt[tgt == -1] = 89
    return x, tgt

class EPianoDataset(data.Dataset):
    def __init__(self, root, max_seq = 1024, random_seq = True):
        self.root = root
        self.max_seq = max_seq
        self.random_seq = random_seq

        fs = [os.path.join(root, f) for f in os.listdir(self.root)]
        self.data_files = [f for f in fs if os.path.isfile(f)]
    
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        i_stream = open(self.data_files[idx], 'rb')
        raw_mid = torch.tensor(pickle.load(i_stream), dtype = torch.long)
        # print(raw_mid.shape)
        i_stream.close()

        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

        return {'input_ids': x, 'labels': tgt}
    
def create_epiano_datasets(dataset_root, max_seq, random_seq = True):
    train_root = os.path.join(dataset_root, 'train')
    val_root = os.path.join(dataset_root, 'val')
    test_root = os.path.join(dataset_root, 'test')

    train_dataset = EPianoDataset(train_root, max_seq, random_seq)
    val_dataset = EPianoDataset(val_root, max_seq, random_seq)
    test_dataset = EPianoDataset(test_root, max_seq, random_seq)

    return train_dataset, val_dataset, test_dataset

def create_epiano_dataloaders(input_dir, max_seq, batch_size, num_workers, random_seq = True):
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(input_dir, max_seq, random_seq)
    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    val_loader = data.DataLoader(val_dataset, batch_size = 32, num_workers = num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size = 2, num_workers = num_workers)
    return train_loader, val_loader, test_loader

def compute_epiano_accuracy(out, tgt):
    softmax = nn.Softmax(dim = -1)
    out = torch.argmax(softmax(out), dim = -1)

    out = out.flatten()
    tgt = tgt.flatten()
    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(torch.float32)
    acc = num_right / len(tgt)
    return acc

def jsb_refactor_json(file_in, file_out):
    with open(file_in, 'rb') as file:
            bach_chorales = json.load(file)

    new_dataset = {
        'test' : [],
        'train' : [],
        'valid' : []
    }

    for group in list(new_dataset.keys()):
        for chorale in bach_chorales[group]:
            new_chorale = []
            for voices in chorale:
                new_chorale += voices
            new_dataset[group] += [new_chorale]

    with open(file_out, 'w') as file:
        json.dump(new_dataset, file, indent=4)

    return new_dataset 

def make_note_mapping(new_dataset):
    print(new_dataset)

class JSBChoralesDataset(data.Dataset):
    def __init__(self, data_split, max_seq = 1024, random_seq = True):
        self.data_split = data_split
        self.max_seq = max_seq
        self.random_seq = random_seq
    
    def __len__(self):
        return len(self.data_split)
    
    def __getitem__(self, idx):
        output_mid = torch.tensor(self.data_split[idx])
        x, tgt = process_jsb(output_mid, self.max_seq, self.random_seq)
        return {'input_ids': x, 'labels': tgt}

def jsb_chorales_dataloaders(json_file_path, max_seq = 1024, random_seq = True, batch_size = 64, num_workers = 16):
    new_dataset = jsb_refactor_json(json_file_path, 'music-gen/JSB-Chorales-dataset/Jsb16thSeparated_refactored.json')
    train_dataset = JSBChoralesDataset(new_dataset['train'], max_seq, random_seq)
    valid_dataset = JSBChoralesDataset(new_dataset['valid'], max_seq, random_seq)
    test_dataset = JSBChoralesDataset(new_dataset['test'], max_seq, random_seq)

    train_loader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    valid_loader = data.DataLoader(valid_dataset, batch_size = 32, num_workers = num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size = 2, num_workers = num_workers)
    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    json_file_path = 'music-gen/JSB-Chorales-dataset/Jsb16thSeparated.json'
    new_dataset = jsb_refactor_json(json_file_path, 'music-gen/JSB-Chorales-dataset/Jsb16thSeparated_refactored.json')

    train_dataset = new_dataset['valid']
    flat_list = [num for sublist in train_dataset for num in sublist]
    unique_numbers = set(flat_list)
    print(unique_numbers)
    count_unique = len(unique_numbers)
    print(count_unique)

    lengths = {len(sublist) for sublist in new_dataset['train']}
    print(lengths)