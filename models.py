import torch
from torch import autograd, nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers import *
from options import opt

from transformers import BertPreTrainedModel, BertModel, RobertaModel, RobertaConfig, XLMRobertaConfig


class RobertaClassificationHead(nn.Module):

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.lang_discriminator = RobertaClassificationHead(config, 1)
        self.triple_validator = RobertaClassificationHead(config, 2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, _type="ld"):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[0]  # CLS vector

        if _type == "ld":
            logits = self.lang_discriminator(pooled_output)
        elif _type == "tv":
            logits = self.triple_validator(pooled_output)
        else:
            raise ValueError("Wrong Type!")

        outputs = (logits, pooled_output) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # logits, (hidden_states), (attentions)


class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    config_class = XLMRobertaConfig


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lang_discriminator = nn.Linear(config.hidden_size, 1)
        self.triple_validator = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, _type="ld"):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]  # CLS vector
        pooled_output = self.dropout(pooled_output)

        if _type == "ld":
            logits = self.lang_discriminator(pooled_output)
        elif _type == "tv":
            logits = self.triple_validator(pooled_output)
        else:
            raise ValueError("Wrong Type!")

        outputs = (logits, pooled_output) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # logits, (hidden_states), (attentions)


class DANFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 sum_pooling,
                 dropout,
                 batch_norm=False):
        super(DANFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()

        if sum_pooling:
            self.avg = SummingLayer(self.word_emb)
        else:
            self.avg = AveragingLayer(self.word_emb)

        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(vocab.emb_size, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, input):
        return self.fcnet(self.avg(input))


class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 bdrnn,
                 attn_type):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.bdrnn = bdrnn
        self.attn_type = attn_type
        self.hidden_size = hidden_size//2 if bdrnn else hidden_size
        self.n_cells = self.num_layers*2 if bdrnn else self.num_layers

        self.word_emb = vocab.init_embed_layer()
        self.rnn = nn.LSTM(input_size=vocab.emb_size, hidden_size=self.hidden_size,
                num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)
        if attn_type == 'dot':
            self.attn = DotAttentionLayer(hidden_size)

    def forward(self, input):
        data, lengths = input
        lengths_list = lengths.tolist()
        batch_size = len(data)
        embeds = self.word_emb(data)
        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = embeds.data.new(*state_shape)
        output, (ht, ct) = self.rnn(packed, (h0, c0))

        if self.attn_type == 'last':
            return ht[-1] if not self.bdrnn \
                          else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        elif self.attn_type == 'avg':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return torch.sum(unpacked_output, 1) / lengths.float().view(-1, 1)
        elif self.attn_type == 'dot':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return self.attn((unpacked_output, lengths))
        else:
            raise Exception('Please specify valid attention (pooling) mechanism')


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])
        
        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                        nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, input):
        data, lengths = input
        batch_size = len(data)
        embeds = self.word_emb(data)
        # conv
        embeds = embeds.unsqueeze(1) # batch_size, 1, seq_len, emb_size
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)


class TripleValidator(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(TripleValidator, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class LanguageDetector(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm=False):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, 1))

    def forward(self, input):
        return self.net(input)


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False,
                 bert_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        # self.linear = nn.Linear(1, 1)

        # self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.gamma = torch.Tensor([gamma])

        if bert_embedding is False:
            self.embedding_range = nn.Parameter(
            # embedding_range = nn.Parameter(
                torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
                requires_grad=False
            )

            self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
            self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
                #a=-embedding_range.item(),
                #b=embedding_range.item()
            )

            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.relation_embedding,
                a=-self.embedding_range.item(),
                #a=-embedding_range.item(),
                b=self.embedding_range.item()
                #b=embedding_range.item()
            )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

    def forward(self, sample, mode='single', bert_embedding=False):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            print(mode)
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            # if bert_embedding:
            #     head = self.head_dense(head)
            #     relation = self.relation_dense(relation)
            #     tail = self.tail_dense(tail)

            score = model_func[self.model_name](head, relation, tail, mode)
            # score = self.linear(score.unsqueeze(2)).squeeze(dim=2)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            # head = (N, negative_size, 500), relation = (N, 1, 500), tail = (N, 1, 500)
            score = head + (relation - tail)  # score = (N, negative_size, 500)
        else:
            # tail-batch
            # head = (n, 1, 500), relation = (n, 1, 500), tail = (n, negative_size, 500)
            # single
            # head = (N, 1, 500), relation = (N, 1, 500), tail = (N, 1, 500)
            score = (head + relation) - tail

        # train => tail-batch (N, negative_size, 768) or single: (N, 1, 768)
        # test  => (N/2, 40943)
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, device):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        positive_sample = positive_sample.to(device)
        negative_sample = negative_sample.to(device)
        subsampling_weight = subsampling_weight.to(device)

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, nentity, nrelation, args, device):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                nentity,
                nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                nentity,
                nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    positive_sample = positive_sample.to(device)
                    negative_sample = negative_sample.to(device)
                    filter_bias = filter_bias.to(device)

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % 1000 == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
