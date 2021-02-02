import argparse
import logging
import os
import random
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

from datasets import KGProcessor
from models import *
from options import opt
from vocab import Vocab
import utils

from transformers import BertTokenizer, AdamW, XLMRobertaTokenizer
from torch.utils.tensorboard import SummaryWriter

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# save logs
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr,
                    format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)

# output options
log.info('Training M-BERT with options:')
log.info(opt)

mse_loss = nn.MSELoss()

def train(opt):
    target = opt.target_lang
    summary = SummaryWriter(log_dir=opt.tb_log_dir)

    device = torch.device(opt.device)
    n_gpu = torch.cuda.device_count()
    if device == torch.device("cuda"):
        log.info("device: {} n_gpu: {} ".format(device, n_gpu))
    else:
        log.info("device: {}".format("cpu"))

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(opt.seed)

    # processor
    log.info(f'Loading data processor...')

    if opt.bert_model.split("-")[0] == "bert":
        tokenizer = BertTokenizer.from_pretrained(opt.bert_model, do_lower_case=opt.do_lower_case)
    else:
        tokenizer = XLMRobertaTokenizer.from_pretrained(opt.bert_model)
    processor = KGProcessor(tokenizer)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # datasets
    if opt.do_train and not opt.monolingual:
        log.info(f'Loading en data...')
        if opt.tv_train:
            en_train_filename = os.path.join(opt.src_data_dir, opt.tv_train)
        else:
            en_train_filename = os.path.join(opt.src_data_dir, 'en_train.tsv')
        en_dev_filename = os.path.join(opt.src_data_dir, 'en_dev.tsv')
        en_train = processor.get_tensor_dataset(opt, en_train_filename, log)
        en_dev = processor.get_tensor_dataset(opt, en_dev_filename, log)

        log.info(f'Loading kr data...')
        kr_train_filename = os.path.join(opt.tgt_data_dir, '%s_train.tsv' % target)
        if opt.dev_file:
            kr_dev_filename = os.path.join(opt.tgt_data_dir, opt.dev_file)
        else:
            kr_dev_filename = os.path.join(opt.tgt_data_dir, '%s_dev.tsv' % target)
        kr_train = processor.get_tensor_dataset(opt, kr_train_filename, log)
        kr_dev = processor.get_tensor_dataset(opt, kr_dev_filename, log)
    elif opt.monolingual:
        kr_train_filename = os.path.join(opt.tgt_data_dir, opt.mono_train)
        if opt.dev_file:
            kr_dev_filename = os.path.join(opt.tgt_data_dir, opt.dev_file)
        else:
            kr_dev_filename = os.path.join(opt.tgt_data_dir, '%s_dev.tsv' % target)
        kr_train = processor.get_tensor_dataset(opt, kr_train_filename, log)
        kr_dev = processor.get_tensor_dataset(opt, kr_dev_filename, log)
        en_train = kr_train
        en_dev = kr_dev

    if opt.test_file:
        kr_test_filename = os.path.join(opt.tgt_data_dir, opt.test_file)
    else:
        if target == "kr":
            kr_test_filename = os.path.join(opt.tgt_data_dir, '%s_test.tsv' % target)
        else:
            kr_test_filename = os.path.join(opt.tgt_data_dir, '%s_test_easy.tsv' % target)
    kr_test = processor.get_tensor_dataset(opt, kr_test_filename, log)
    log.info("Loading test file with %s" % kr_test_filename)
    log.info('Done loading datasets.')
    opt.num_labels = 2

    # if pretrained vec file too big, save reduced word vectors.
    # vocab.save_reduced_vec()

    if opt.do_train and not opt.monolingual:
        en_train_loader = DataLoader(en_train, opt.batch_size, shuffle=True)
        en_train_loader_Q = DataLoader(en_train, opt.batch_size, shuffle=True)
        kr_train_loader = DataLoader(kr_train, opt.batch_size, shuffle=True)
        kr_train_loader_Q = DataLoader(kr_train, opt.batch_size, shuffle=True)
        en_train_iter_Q = iter(en_train_loader_Q)
        kr_train_iter = iter(kr_train_loader)
        kr_train_iter_Q = iter(kr_train_loader_Q)

        en_dev_loader = DataLoader(en_dev, opt.batch_size, shuffle=False)
        kr_dev_loader = DataLoader(kr_dev, opt.batch_size, shuffle=False)
    elif opt.monolingual:
        en_train_loader = DataLoader(en_train, opt.batch_size, shuffle=True)
        en_dev_loader = DataLoader(en_dev, opt.batch_size, shuffle=False)

    kr_test_loader = DataLoader(kr_test, opt.batch_size, shuffle=False)

    # models
    if opt.bert_model.split("-")[0] == "bert":
        model = BertForSequenceClassification.from_pretrained(opt.bert_model, num_labels=num_labels)
    else:
        model = XLMRobertaForSequenceClassification.from_pretrained(opt.bert_model)
    model.to(device)

    if opt.do_train:
        # optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate)

        # training
        best_f1 = 0.0
        best_hits1 = 0.0
        global_step = 0
        for epoch in range(opt.max_epoch):
            model.train()
            en_train_iter = iter(en_train_loader)
            # training accuracy
            correct, total = 0, 0
            for i, en_tv_batch in tqdm(enumerate(en_train_iter),
                                                   total=len(en_train)//opt.batch_size):
                if not opt.monolingual:
                    try:
                        inputs_kr = next(kr_train_iter)  # Korean labels are not used
                    except:
                        # check if Korean data is exhausted
                        kr_train_iter = iter(kr_train_loader)
                        inputs_kr = next(kr_train_iter)

                # Q iterations: train Language Discriminator Q
                n_critic = opt.n_critic
                if opt.monolingual:
                    n_critic = 0
                    opt.n_critic = 0
                if n_critic > 0 and ((epoch==0 and i<=25) or (i%500==0)):
                    n_critic = 3
                for qiter in range(n_critic):
                    # clip Q weights
                    # TODO: cliping..?
                    for p in model.parameters():
                        p.data.clamp_(opt.clip_lower, opt.clip_upper)
                    for p in model.lang_discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)
                    optimizer.zero_grad()
                    # get a minibatch of data
                    try:
                        # labels are not used
                        q_inputs_en = next(en_train_iter_Q)
                    except StopIteration:
                        # check if dataloader is exhausted
                        en_train_iter_Q = iter(en_train_loader_Q)
                        q_inputs_en = next(en_train_iter_Q)
                    try:
                        q_inputs_kr = next(kr_train_iter_Q)
                    except StopIteration:
                        kr_train_iter_Q = iter(kr_train_loader_Q)
                        q_inputs_kr = next(kr_train_iter_Q)

                    en_input_ids, en_input_mask, en_segment_ids, en_label = (t.to(opt.device) for t in q_inputs_en)
                    __q_outputs_en = model(en_input_ids,
                                           token_type_ids=en_segment_ids,
                                           attention_mask=en_input_mask, _type="ld")
                    #loss_fct = nn.CrossEntropyLoss()
                    #q_en_loss = loss_fct(q_outputs_en[0], en_label.new_ones(en_label.size()))
                    en_logit = __q_outputs_en[0]
                    #en_p = nn.Sigmoid()(en_logit)
                    q_outputs_en = en_logit.mean()
                    #(-q_outputs_en).backward()

                    kr_input_ids, kr_input_mask, kr_segment_ids, kr_label = (t.to(opt.device) for t in q_inputs_kr)
                    __q_outputs_kr = model(kr_input_ids,
                                           token_type_ids=kr_segment_ids,
                                           attention_mask=kr_input_mask, _type="ld")
                    #loss_fct = nn.CrossEntropyLoss()
                    #q_kr_loss = loss_fct(q_outputs_kr[0], kr_label.new_zeros(kr_label.size()))
                    kr_logit = __q_outputs_kr[0]
                    #kr_p = nn.Sigmoid()(kr_logit)
                    q_outputs_kr = kr_logit.mean()
                    #q_outputs_kr.backward()
                    (-q_outputs_en + q_outputs_kr).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                # clip Q weights
                # TODO cliping..?
                for p in model.parameters():
                        p.data.clamp_(opt.clip_lower, opt.clip_upper)
                for p in model.lang_discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                #F.zero_grad()
                #P.zero_grad()

                optimizer.zero_grad()

                # 1) train F and P
                en_input_ids, en_input_mask, en_segment_ids, en_label = (t.to(opt.device) for t in en_tv_batch)
                tv_outputs_en = model(en_input_ids,
                                      token_type_ids=en_segment_ids,
                                      attention_mask=en_input_mask, _type="tv")
                loss_fct = nn.CrossEntropyLoss()
                q_en_loss = loss_fct(tv_outputs_en[0], en_label)
                q_en_loss.backward()

                # training accuracy of triple validator P on en triple set
                _, pred = torch.max(tv_outputs_en[0], dim=1)
                total += en_label.size(0)
                correct += (pred == en_label).sum().item()

                if opt.n_critic > 0:  # with adversarial training
                    # 2) train F (not Q) with en data
                    __ld_outputs_en = model(en_input_ids,
                                            token_type_ids=en_segment_ids,
                                            attention_mask=en_input_mask, _type="ld")

                    ld_outputs_en = __ld_outputs_en[0].mean()
                    (opt.lambd * ld_outputs_en).backward()

                    # 3) train F (not Q) with kr data for learning language-invariant features.
                    # ??? W(P_F_en, P_F_kr) = E[g(f(x_en))] - E[g(f(x_kr))]
                    kr_input_ids, kr_input_mask, kr_segment_ids, _ = (t.to(opt.device) for t in inputs_kr)
                    __ld_outputs_kr = model(kr_input_ids,
                                            token_type_ids=kr_segment_ids,
                                            attention_mask=kr_input_mask, _type="ld")
                    ld_outputs_kr = __ld_outputs_kr[0].mean()
                    (-opt.lambd * ld_outputs_kr).backward()

                optimizer.step()

                if global_step % 20 == 0:
                    log.info("step: %d, n_critic: %d" % (global_step, n_critic))
                    if opt.n_critic > 0:
                        summary.add_scalar('loss_lang_discriminator_en', (-q_outputs_en).item(), global_step)
                        summary.add_scalar('loss_lang_discriminator_tgt', q_outputs_kr.item(), global_step)
                        summary.add_scalar('p_en_', ld_outputs_en.item(), global_step)
                        summary.add_scalar('p_kr_', (-ld_outputs_kr).item(), global_step)
                        summary.add_scalar('MSE_en_tgt', mse_loss(__ld_outputs_en[0], __ld_outputs_kr[0]), global_step)
                    summary.add_scalar('loss_en_triple_vaildator', q_en_loss.item(), global_step)
                    summary.add_scalar('train_accurary_en_triple_validator', (correct/total), global_step)
                global_step += 1

                if global_step % 200 == 0:
                    if opt.monolingual:
                        kr_dev_loader = en_dev_loader
                    #_ = evaluate(opt, kr_dev_loader, model, "dev", target)
                    _ = evaluate(opt, kr_test_loader, model, "test", target)

                if global_step > opt.max_step:
                    break
            ############################
            # end of epoch
            log.info('Ending epoch {}'.format(epoch+1))
            # evaluate
            log.info('Training Accuracy: {}%'.format(100.0*correct/total))

            if not opt.monolingual:
                log.info('Evaluating English Validation set:')
                _ = evaluate(opt, en_dev_loader, model)

            log.info('Evaluating Foreign validation set:')
            if opt.monolingual:
                kr_dev_loader = en_dev_loader
            result = evaluate(opt, kr_dev_loader, model, "dev", target)
            summary.add_scalar('dev_accuracy_tgt_triple_validator', result["acc"], epoch+1)
            summary.add_scalar('dev_precision_tgt_triple_validator', result["precision"], epoch+1)
            summary.add_scalar('dev_recall_tgt_triple_validator', result["recall"], epoch+1)
            if target == "kr":
                summary.add_scalar('dev_MAP_tgt_triple_validator', result["hits@1"], epoch+1)
            else:
                summary.add_scalar('dev_f1_tgt_triple_validator', result["f1"], epoch+1)
            #if result["f1"] > best_f1:
            if result["hits@1"] > best_hits1:
                log.info(f'New Best Foreign validation f1: {result["hits@1"]}')
                # best_f1 = result["f1"]
                best_hits1 = result["hits@1"]
                best_ep = epoch + 1
            torch.save(model.state_dict(),
                       '{}/pytorch_model_{}.pth'.format(opt.model_save_file, epoch + 1))
            # log.info('Evaluating Foreign test set:')
            # evaluate(opt, kr_test_loader, F, P)
        # log.info(f'Best Foreign validation accuracy: {best_acc} at {best_ep}')
        #log.info(f'Best Foreign validation f1: {best_f1} at {best_ep}')
        log.info(f'Best Foreign validation f1: {best_hits1} at {best_ep}')

    if opt.do_eval:
        log.info('Evaluating Foreign test set:')
        if opt.do_train:
            model.load_state_dict(torch.load('{}/pytorch_model_{}.pth'.format(opt.model_save_file, best_ep)))
        elif opt.do_raw_mbert:
            pass
        else:
            model.load_state_dict(torch.load(opt.test_model))
        _ = evaluate(opt, kr_test_loader, model, "test", target)

    if opt.do_aug:
        model.load_state_dict(torch.load(opt.test_model))
        model.eval()
        it = iter(kr_test_loader)
        correct = 0
        total = 0
        rel = 0
        label_list, logit_list = [], []
        pred_list = []
        with torch.no_grad():
            for batch in tqdm(it):
                input_ids, input_mask, segment_ids, label = (t.to(opt.device) for t in batch)
                tv_outputs = model(input_ids,
                                   token_type_ids=segment_ids,
                                   attention_mask=input_mask, _type="tv")
                _, pred = torch.max(tv_outputs[0], dim=1)
                total += label.size(0)
                logit_list.extend(tv_outputs[0][:, 1].tolist())
                pred_list.extend(pred.tolist())

        answer_filename = os.path.join(opt.tgt_data_dir, "kr_aug_answer.tsv")
        en_triple_list = []
        kr_triple_list = []
        with open(answer_filename, "r") as f:
            for line in f.readlines():
                tokens = [x.strip() for x in line.rstrip().split("\t")]
                en_triple_list.append(tokens[0])
                kr_triple_list.append(line.strip())
        assert len(en_triple_list) == len(pred_list)
        pre_id = en_triple_list[0]
        ranks = 0
        local_logit, local_pred = [], []
        local_kr = []
        top1 = 0
        top3 = 0
        top_n = [[] for _ in range(10)]

        f = open("augmented_kr_triples.txt", "w")
        f2 = open("augmented_kr_triples_highconf.txt", "w")
        for i, pred in enumerate(pred_list):
            cur_id = en_triple_list[i]
            if i > 0 and (cur_id != pre_id):
                ranks += 1
                value, argsort = torch.sort(torch.tensor(local_logit), descending=True)
                argsort = argsort.cpu().numpy().tolist()
                for j, k in enumerate(argsort[:10]):
                    try:
                        if local_pred[k] == 1:
                            top_n[j].append(1)
                            if j == 0:
                                f.write(local_kr[k] + "\n")
                                if local_logit[k] > 1.0:
                                    f2.write(local_kr[k] + "\n")
                    except:
                        pass
                if len(logit_list) < 3:
                    top1 += 1
                else:
                    top1 += 1
                    top3 += 1
                local_logit = []
                local_pred = []
                local_kr = []
            local_logit.append(logit_list[i])
            local_pred.append(pred_list[i])
            local_kr.append(kr_triple_list[i])
            pre_id = cur_id
        ranks += 1
        value, argsort = torch.sort(torch.tensor(local_logit), descending=True)
        argsort = argsort.cpu().numpy().tolist()
        for j, k in enumerate(argsort[:10]):
            try:
                if local_pred[k] == 1:
                    top_n[j].append(1)
                    if j == 0:
                        f.write(local_kr[k] + "\n")
            except:
                pass
        if len(logit_list) < 3:
            top1 += 1
        else:
            top1 += 1
            top3 += 1
        log.info('total: %d, top-1: %d/%d, top-3: %d/%d' % (len(pred_list), top1, ranks, top3, ranks))
        temp = 0
        for e in range(10):
            temp += np.array(top_n[e]).sum()
            print(np.array(top_n[e]).sum(), temp)
        f.close()
        f2.close()


def evaluate(opt, loader, model, type="dev", target_lang="en"):
    model.eval()
    it = iter(loader)
    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0
    rel = 0
    label_list, logit_list = [], []
    pred_list = []
    with torch.no_grad():
        for batch in tqdm(it):
            input_ids, input_mask, segment_ids, label = (t.to(opt.device) for t in batch)
            tv_outputs = model(input_ids,
                               token_type_ids=segment_ids,
                               attention_mask=input_mask, _type="tv")
            _, pred = torch.max(tv_outputs[0], dim=1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            _tp = ((pred == label) & (pred == 1)).sum().item()
            tp += _tp
            fp += ((pred != label) & (pred == 1)).sum().item()
            fn += ((label == 1).sum().item() - _tp)
            rel += (label == 1).sum().item()
            label_list.extend(label.tolist())
            logit_list.extend(tv_outputs[0][:, 1].tolist())
            pred_list.extend(pred.tolist())
    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    log.info('On %d samples: acc: %.4f, precision: %.4f, recall: %.4f, f1-score: %.4f' % (total, accuracy, precision, recall, f1))
    log.info('total: %d, tp: %d, fn: %d, (tp+fn): %d' % (total, tp, fn, (tp+fn)))

    # ranking evaluation
    assert len(label_list) == len(logit_list)
    mrr, n_candis = 0, 0
    temp, arg_list, ranks = [], [], []

    if target_lang == "kr":
        tagged_filename = os.path.join(opt.tgt_data_dir, "kr_origin_%s.csv" % type)
        en_triple_list = []
        with open(tagged_filename, "r") as f:
            for line in f.readlines():
                tokens = [x.strip() for x in line.rstrip().split(",")]
                e_h, e_r, e_t = tokens[:3]
                en_triple_list.append(e_h+e_r+e_t)
        local_logits, local_labels = [], []
        local_label_sum = 0
        pre_id = en_triple_list[0]
        map_list, rank_list = [], []
        all_zero, all_ranks = 0, 0
        hit1, hit3 = [], []
        for i, label in enumerate(label_list):
            cur_id = en_triple_list[i]
            if i > 0 and (cur_id != pre_id):
                all_ranks += 1
                if local_label_sum > 0:
                    value, argsort = torch.sort(torch.tensor(local_logits), descending=True)
                    argsort = argsort.cpu().numpy()
                    local_map = 0
                    n_candis += len(local_logits)
                    for j, k in enumerate(argsort.tolist()):
                        arg_list.append(np.where(argsort == j)[0][0] + 1)
                        if local_labels[k] == 1:
                            local_map += 1/(j+1)
                            rank_list.append(j+1)
                            if j == 0:
                                hit1.append(1)
                    for j, k in enumerate(argsort.tolist()):
                        if local_labels[k] == 1 and j < 3:
                            hit3.append(1)
                            break
                    map_list.append((local_map/local_label_sum))
                    local_logits, local_labels = [], []
                    local_label_sum = 0
                else:  # local_label_sum = 0
                    value, argsort = torch.sort(torch.tensor(local_logits), descending=True)
                    argsort = argsort.cpu().numpy()
                    for j, k in enumerate(argsort.tolist()):
                        arg_list.append(np.where(argsort == j)[0][0] + 1)
                    all_zero += 1
                    local_logits, local_labels = [], []
                    local_label_sum = 0
            local_logits.append(logit_list[i])
            local_label_sum += label
            local_labels.append(label)
            pre_id = cur_id
        all_ranks += 1
        if local_label_sum == 0:
            all_zero += 1
            value, argsort = torch.sort(torch.tensor(local_logits), descending=True)
            argsort = argsort.cpu().numpy()
            for j, k in enumerate(argsort.tolist()):
                arg_list.append(np.where(argsort == j)[0][0] + 1)
        else:
            value, argsort = torch.sort(torch.tensor(local_logits), descending=True)
            argsort = argsort.cpu().numpy()
            local_map = 0
            n_candis += len(local_logits)
            for j, k in enumerate(argsort.tolist()):
                arg_list.append(np.where(argsort == j)[0][0] + 1)
                if local_labels[k] == 1:
                    local_map += 1/(j+1)
                    rank_list.append(j+1)
                    if j == 0:
                        hit1.append(1)
            for j, k in enumerate(argsort.tolist()):
                if local_labels[k] == 1 and j < 3:
                    hit3.append(1)
                    break
            map_list.append((local_map / local_label_sum))
        log.info('On %d ranking test: MAP: %.4f, MR: %.4f  (all zeros: %d/%d) (avg candidates: %.4f)' % (
            len(map_list), np.array(map_list).mean(), np.array(rank_list).mean(), all_zero, all_ranks, n_candis/len(map_list)))
        log.info("hits@1: %.4f, hits@3: %.4f" % (np.array(hit1).sum()/len(map_list), np.array(hit3).sum()/len(map_list)))
        result = {"acc": accuracy, "precision": precision, "recall": recall, "f1": f1, "MAP": np.array(map_list).mean(),
                  "hits@1": np.array(hit1).sum()/len(map_list)}
    else:  # other language
        top_n_true = [0 for _ in range(10)]
        prec_n = [[] for _ in range(10)]
        acc_n = [[] for _ in range(10)]
        local_label, local_pred = [], []
        for i, label in enumerate(label_list):
            if i != 0 and label == 1:  # go to next
                value, argsort = torch.sort(torch.tensor(temp), descending=True)
                argsort = argsort.cpu().numpy()
                for j, k in enumerate(argsort.tolist()):
                    arg_list.append(np.where(argsort == j)[0][0] + 1)
                for j, k in enumerate(argsort.tolist()[:10]):
                    if local_label[k] == 1:
                        top_n_true[j] += 1
                local_correct, local_tp, local_pos = 0, 0, 0
                for j in range(10):
                    if j < len(argsort):
                        k = argsort[j]
                        if local_label[k] == local_pred[k]:
                            local_correct += 1
                        if local_label[k] == 1 and local_pred[k] == 1:
                            local_tp += 1
                        if local_pred[k] == 1:
                            local_pos += 1
                        _prec = 0 if local_pos == 0 else local_tp / local_pos
                        prec_n[j].append(_prec)
                        acc_n[j].append(local_correct / (j + 1))
                    else:
                        _prec = 0 if local_pos == 0 else local_tp / local_pos
                        prec_n[j].append(_prec)
                        acc_n[j].append(local_correct / len(argsort))
                rank = np.where(argsort == 0)[0][0] + 1
                ranks.append(rank)
                mrr += 1/rank
                n_candis += len(temp)
                temp = []
                local_label, local_pred = [], []
            temp.append(logit_list[i])
            local_label.append(label)
            local_pred.append(pred_list[i])
        value, argsort = torch.sort(torch.tensor(temp), descending=True)
        argsort = argsort.cpu().numpy()
        for j, k in enumerate(argsort.tolist()):
            arg_list.append(np.where(argsort == j)[0][0] + 1)
        for j, k in enumerate(argsort.tolist()[:10]):
            if local_label[k] == 1:
                top_n_true[j] += 1
        local_correct, local_tp, local_pos = 0, 0, 0
        for j in range(10):
            try:
                k = argsort[j]
                if local_label[k] == local_pred[k]:
                    local_correct += 1
                if local_label[k] == 1 and local_pred[k] == 1:
                    local_tp += 1
                if local_pred[k] == 1:
                    local_pos += 1
                _prec = 0 if local_pos == 0 else local_tp/local_pos
                prec_n[j].append(_prec)
                acc_n[j].append(local_correct / (j + 1))
            except:
                _prec = 0 if local_pos == 0 else local_tp/local_pos
                prec_n[j].append(_prec)
                acc_n[j].append(local_correct / len(argsort))
        rank = np.where(argsort == 0)[0][0] + 1
        ranks.append(rank)
        mrr += 1/rank
        assert rel == len(ranks)
        log.info('On %d ranking test: MRR: %.4f, MR: %.4f' % (len(ranks), mrr/len(ranks), np.array(ranks).mean()))
        hits1, hits3, hits5 = 0, 0, 0
        for r in ranks:
            if r <= 5:
                hits5 += 1
            if r <= 3:
                hits3 += 1
            if r == 1:
                hits1 += 1
        log.info('Hits@1: %.4f, Hits@3: %.4f, Hits@5: %.4f (avg candidates: %.4f)' % (
            hits1/len(ranks), hits3/len(ranks), hits5/len(ranks), (n_candis/len(ranks))))
        result = {"acc": accuracy, "precision": precision, "recall": recall, "f1": f1, "hits@1": hits1/len(ranks)}

        log.info("%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f" % (
            accuracy, precision, recall, f1, mrr/len(ranks), np.array(ranks).mean(),
            hits1/len(ranks), hits3/len(ranks), hits5/len(ranks),
        ))
        top_acc = []
        tp = 0
        for x in range(10):
            tp += top_n_true[x]
            if len(ranks) > 0:
                top_acc.append(tp/(len(ranks)*(x+1)))
            else:
                top_acc.append(0)
        print(" ".join([str(round(x, 4)) for x in top_acc]))
        final_acc_n = []
        for x in range(10):
            final_acc_n.append(np.array(acc_n[x]).mean())
        print("Accuracy@N: " + " ".join([str(round(x, 4)) for x in final_acc_n]))

    # for debugging
    if opt.debugging or target_lang == "kr":
        if type == "dev":
            dev_filename = os.path.join(opt.tgt_data_dir, opt.dev_file)
        else:
            dev_filename = os.path.join(opt.tgt_data_dir, opt.test_file)
        debug_file = "debugging_result.txt"
        with open(dev_filename, "r") as f:
            dev_data = [x.strip() for x in f.readlines()]
        assert len(dev_data) == len(logit_list)
        assert len(dev_data) == len(arg_list)
        assert len(dev_data) == len(pred_list)

        pre_id = en_triple_list[0]
        local_logits, local_labels = [], []
        local_label_sum = 0
        hit1, hit3 = [], []
        ranks = 0
        top_n_true = [0 for _ in range(10)]
        map_list = []
        prec_n = [[] for _ in range(10)]
        for i, pred in enumerate(pred_list):
            cur_id = en_triple_list[i]
            if i > 0 and (cur_id != pre_id):
                if local_label_sum > 0:
                    ranks += 1
                    # print(pre_id, local_logits)
                    assert len(local_labels) == len(local_logits)
                    value, argsort = torch.sort(torch.tensor(local_logits), descending=True)
                    argsort = argsort.cpu().numpy().tolist()
                    local_map = 0
                    for j, k in enumerate(argsort):
                        if local_labels[k] == 1:
                            local_map += 1/(j+1)
                    map_list.append((local_map/local_label_sum))
                    for j, k in enumerate(argsort):
                        if j == 0 and local_labels[k] == 1:
                            hit1.append(1)
                            break
                    for j, k in enumerate(argsort):
                        if local_labels[k] == 1 and j < 3:
                            hit3.append(1)
                            break
                    _local_sum = 0
                    for j, k in enumerate(argsort[:10]):
                        if local_labels[k] == 1:
                            top_n_true[j] += 1
                    local_tp = 0
                    for j in range(10):
                        try:
                            k = argsort[j]
                            if local_labels[k] == 1:
                                local_tp += 1
                            prec_n[j].append(local_tp/(j+1))
                        except:
                            prec_n[j].append(local_tp/len(argsort))
                local_logits, local_labels = [], []
                local_label_sum = 0
            else:
                if pred == 1:
                    local_logits.append(logit_list[i])
                    local_label_sum += label_list[i]
                    local_labels.append(label_list[i])
            pre_id = cur_id
        if local_label_sum > 0:
            ranks += 1
            value, argsort = torch.sort(torch.tensor(local_logits), descending=True)
            argsort = argsort.cpu().numpy()
            local_map = 0
            for j, k in enumerate(argsort.tolist()):
                if local_labels[k] == 1:
                    local_map += 1/(j+1)
            map_list.append((local_map/local_label_sum))
            for j, k in enumerate(argsort.tolist()):
                if local_labels[k] == 1 and j == 0:
                    hit1.append(1)
                    break
            for j, k in enumerate(argsort.tolist()):
                if local_labels[k] == 1 and j < 3:
                    hit3.append(1)
                    break
            _local_sum = 0
            for j, k in enumerate(argsort[:10]):
                if local_labels[k] == 1:
                    top_n_true[j] += 1
            local_tp = 0
            for j in range(10):
                try:
                    k = argsort[j]
                    if local_labels[k] == 1:
                        local_tp += 1
                    prec_n[j].append(local_tp / (j + 1))
                except:
                    prec_n[j].append(local_tp / len(argsort))
        if len(hit1) == 0:
            hit1 = [0]
        if len(hit3) == 0:
            hit3 = [0]
        top_acc = []
        tp = 0
        for x in range(10):
            tp += top_n_true[x]
            if ranks > 0:
                top_acc.append(tp/(ranks*(x+1)))
            else:
                top_acc.append(0)

        assert len(prec_n[0]) == ranks
        final_prec_n = []
        for x in range(10):
            final_prec_n.append(np.array(prec_n[x]).mean())

        print("top_n_true :", top_n_true)
        print("top_n_total:", [ranks*(x+1) for x in range(10)])
        print("top_n_acc  :", " ".join([str(round(x, 4)) for x in top_acc]))
        print("Precision@N: " + " ".join([str(round(x, 4)) for x in final_prec_n]))
        log.info("[two-stage] MAP: %.4f, hits@1: %.4f, hits@3: %.4f" % (
            np.array(map_list).mean(), np.array(hit1).sum()/ranks, np.array(hit3).sum()/ranks))
        with open(debug_file, "w") as fw:
            for i, line in enumerate(dev_data):
                fw.write("%60s\t%d\t%.6f\t%d\n" % (line, pred_list[i], logit_list[i], arg_list[i]))

    return result


if __name__ == '__main__':
    train(opt)

