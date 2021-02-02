import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, TensorDataset
import csv
import sys


class EnTripleDataset(Dataset):
    def __init__(self, data_file_path, vocab, update_vocab):
        self.X = []
        self.Y = []
        with open(data_file_path, "r") as f:
            count = 0
            for line in f:
                # print(line.rstrip().split("\t"))
                triple, label = line.rstrip().split("\t")
                words = triple.strip().split()
                if update_vocab:
                    for w in words:
                        vocab.add_word(w)
                self.X.append(([vocab.lookup(w) for w in words], len(words)))
                self.Y.append(int(label))
                count += 1
        print("English {} triples are loaded from {}".format(len(self.X), data_file_path))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


class TargetTripleDataset(Dataset):
    def __init__(self, opt, data_file_path, vocab, update_vocab, wlabel=False):
        self.X = []
        self.Y = []
        with open(data_file_path, "r") as f:
            count = 0
            for idx, line in enumerate(f):
                if line.find("|||") != -1:
                    a, b, c, label = [x.strip() for x in line.strip().split("|||")]
                    if opt.sov:
                        d = "<s> %s <sep> %s <sep> %s </s>" % (a, c, b)
                    else:
                        d = "<s> %s <sep> %s <sep> %s </s>" % (a, b, c)
                    words = d.split()
                else:
                    if wlabel is True:
                        d, label = line.rstrip().split("\t")
                        if opt.sov:
                            h, r, t = [x.strip() for x in d.split("<sep>")]
                            h = h.split("<s>")[1].strip()
                            t = t.split("</s>")[0].strip()
                            d = "<s> %s <sep> %s <sep> %s </s>" % (h, t, r)
                        words = d.rstrip().split()
                        if int(label) == 2:
                            label = 1
                    else:
                        if opt.sov:
                            h, r, t = [x.strip() for x in line.split("<sep>")]
                            h = h.split("<s>")[1].strip()
                            t = t.split("</s>")[0].strip()
                            line = "<s> %s <sep> %s <sep> %s </s>" % (h, t, r)
                        words = line.rstrip().split()
                        label = 1
                if idx < 3:
                    print(words)

                if update_vocab:
                    for w in words:
                        vocab.add_word(w)
                self.X.append(([vocab.lookup(w) for w in words], len(words)))
                self.Y.append(int(label))
                count += 1
        print("Target {} triples are loaded from {}".format(len(self.X), data_file_path))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def get_en_triple_datasets(vocab, train_filename, dev_filename):
    train = EnTripleDataset(train_filename, vocab, update_vocab=True)
    dev = EnTripleDataset(dev_filename, vocab, update_vocab=True)
    return train, dev

def get_target_triple_datasets(opt, vocab, train_filename, dev_filename, test_filename):
    train = TargetTripleDataset(opt, train_filename, vocab, update_vocab=True)
    dev = TargetTripleDataset(opt, dev_filename, vocab, update_vocab=True, wlabel=True)
    test = TargetTripleDataset(opt, test_filename, vocab, update_vocab=True, wlabel=True)
    return train, dev, test


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""

    def __init__(self, tokenizer):
        self.labels = set()
        self.entities = []
        self.relations = []
        self.num_entity = 0
        self.num_relation = 0
        self.ent2id = {}
        self.rel2id = {}
        self.tokenizer = tokenizer

        # self._set_entities(data_dir)
        # self._set_relations(data_dir)

    def _set_entities(self, data_dir):
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            entities = []
            ent2id = {}
            for i, line in enumerate(f.readlines()):
                ent = line.strip()
                entities.append(ent)
                ent2id[ent] = i
        self.entities = entities
        self.num_entity = len(entities)
        self.ent2id = ent2id

    def _set_relations(self, data_dir):
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            relations = []
            rel2id = {}
            for i, line in enumerate(f.readlines()):
                rel = line.strip()
                relations.append(rel)
                rel2id[rel] = i
        self.relations = relations
        self.num_relation = len(relations)
        self.rel2id = rel2id

    def get_tensor_dataset(self, opt, filename, log):
        lines = self._read_tsv(filename)
        if opt.bert_model.split("-")[0] == "bert":
            features = convert_lines_to_features_bert(opt, lines, self.get_labels(), opt.max_seq_len, self.tokenizer, log)
        else:
            features = convert_lines_to_features_roberta(lines, self.get_labels(), opt.max_seq_len, self.tokenizer, log)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_masks, all_segment_ids, all_label_ids)

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        return self.relations

    def get_labels(self):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        return ["0", "1"]

    def get_entities(self):
        """Gets all entities in the knowledge graph."""
        return self.entities

    def get_entity2text(self, data_dir):
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]
        return ent2text

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._get_triples(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._get_triples(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._get_triples(os.path.join(data_dir, "test.tsv"))

    def _get_triples(self, file_path):
        if len(self.ent2id) == 0:
            self._set_entities()
        triples = []
        with open(file_path) as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                triples.append((self.ent2id[h], self.rel2id[r], self.ent2id[t]))
        return triples

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for i, line in enumerate(ent_lines):
                temp = line.strip().split('\t')
                if len(temp) == 2:
                    end = temp[1]  # .find(',')
                    ent2text[temp[0]] = temp[1]  # [:end]

        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2textlong.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    # first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]  # [:first_sent_end_position + 1]

        entities = list(ent2text.keys())

        rel2text = {}
        with open(os.path.join(data_dir, "relation2text.txt"), 'r') as f:
            rel_lines = f.readlines()
            for i, line in enumerate(rel_lines):
                temp = line.strip().split('\t')
                rel2text[temp[0]] = temp[1]

        lines_str_set = set(['\t'.join(line) for line in lines])
        examples = []
        for (i, line) in enumerate(lines):
            head_id = self.ent2id[line[0]]
            relation_id = self.rel2id[line[1]]
            tail_id = self.ent2id[line[2]]

            head_ent_text = ent2text[line[0]]
            tail_ent_text = ent2text[line[2]]
            relation_text = rel2text[line[1]]

            if set_type == "dev" or set_type == "test":
                label = "1"

                guid = "%s-%s" % (set_type, i)
                self.labels.add(label)
                examples.append(
                    InputExample(guid=guid, head=head_ent_text, relation=relation_text, tail=tail_ent_text,
                                 head_id=head_id, relation_id=relation_id, tail_id=tail_id, label=label))

            elif set_type == "train":
                # true example
                guid = "%s-%s" % (set_type, i)
                examples.append(
                    InputExample(guid=guid, head=head_ent_text, relation=relation_text, tail=tail_ent_text,
                                 head_id=head_id, relation_id=relation_id, tail_id=tail_id, label="1"))

                # corrupted example
                rnd = random.random()
                guid = "%s-%s" % (set_type + "_corrupt", i)
                if rnd <= 0.5:  # corrupting head
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[0])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_head = random.choice(tmp_ent_list)
                            tmp_triple_str = tmp_head + '\t' + line[1] + '\t' + line[2]
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_head_text = ent2text[tmp_head]
                        examples.append(
                            InputExample(guid=guid, head=tmp_head_text, relation=relation_text, tail=tail_ent_text,
                                         head_id=self.ent2id[tmp_head], relation_id=relation_id, tail_id=tail_id, label="0"))
                else:  # corrupting tail
                    for j in range(5):
                        while True:
                            tmp_ent_list = set(entities)
                            tmp_ent_list.remove(line[2])
                            tmp_ent_list = list(tmp_ent_list)
                            tmp_tail = random.choice(tmp_ent_list)
                            tmp_triple_str = line[0] + '\t' + line[1] + '\t' + tmp_tail
                            if tmp_triple_str not in lines_str_set:
                                break
                        tmp_tail_text = ent2text[tmp_tail]
                        examples.append(
                            InputExample(guid=guid, head=head_ent_text, relation=relation_text, tail=tmp_tail_text,
                                         head_id=head_id, relation_id=relation_id, tail_id=self.ent2id[tmp_tail], label="0"))
        return examples


def convert_lines_to_features_bert(opt, examples, label_list, max_seq_length, tokenizer, logger,  print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        if len(example) == 0:
            continue

        try:
            sentence, label = example
            if opt.sov:
                h, r, t = [x.strip() for x in sentence.split("<sep>")]
                h = h.split("<s>")[1].strip()
                t = t.split("</s>")[0].strip()
                sentence = "<s> %s <sep> %s <sep> %s </s>" % (h, t, r)
        except:  # for Korean train data (don't need label)
            sentence = example[0]
            if sentence.find("|||") != -1:
                a, b, c, label = [x.strip() for x in sentence.strip().split("|||")]
                if a.find("/") != -1:
                    a = a.split("/")[-1]
                if b.find("/") != -1:
                    b = b.split("/")[-1]
                if c.find("/") != -1:
                    c = c.split("/")[-1]
                if opt.sov:
                    sentence = "<s> %s <sep> %s <sep> %s </s>" % (a, c, b)
                else:
                    sentence = "<s> %s <sep> %s <sep> %s </s>" % (a, b, c)
            else:
                if opt.sov:
                    h, r, t = [x.strip() for x in sentence.split("<sep>")]
                    h = h.split("<s>")[1].strip()
                    t = t.split("</s>")[0].strip()
                    sentence = "<s> %s <sep> %s <sep> %s </s>" % (h, t, r)
                label = "1"

        sentence = sentence.replace("<s>", "[CLS]").replace("<sep>", "[SEP]").replace("</s>", "[SEP]")

        tokens = tokenizer.tokenize(sentence)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_lines_to_features_roberta(examples, label_list, max_seq_length, tokenizer, logger,  print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        try:
            sentence, label = example
            if label == "2":  # FIXME later ^.^
                label = "1"
        except:  # for Korean train data (don't need label)
            sentence = example[0]
            label = "1"

        sentence = sentence.replace("<sep>", "</s></s>")

        tokens = tokenizer.tokenize(sentence)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        # (c) for sequence triples:
        #  tokens: [CLS] Steve Jobs [SEP] founded [SEP] Apple Inc .[SEP]
        #  type_ids: 0 0 0 0 1 1 0 0 0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence or the third sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()
