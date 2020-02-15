import sys

from typing import *

import copy
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

from torch.utils.tensorboard import SummaryWriter

import pickle

import numpy as np
# from sklearn.metrics import f1_score
import mojimoji

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer, BertConfig
from transformers import BertModel, BertPreTrainedModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import PYTORCH_PRETRAINED_BERT_CACHE

from transformers import BertForNextSentencePrediction
# from transformers import BertForPreTraining

from transformers import add_start_docstrings

from pyknp import Juman


# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logger = logging.getLogger(__name__)

class JumanTokenizer():
    def __init__(self):
        self.juman = Juman(jumanpp=True)

    def tokenize(self, text):
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class MultiInputWrapper(nn.Module):
    def __init__(self, config):
        super(MultiInputWrapper, self).__init__()
        """
        複数入力に対するBERTラッパークラス(2発話想定)
        initではBERTモデルのセットアップ
        callでの複数入力はリストで与えられる想定(そうで無い場合はエラー終了する)
        """
        self.bert_model = BertModel(config)

    def __call__(self, first_input_ids, second_input_ids,
                 first_token_type_ids, second_token_type_ids,
                 first_attention_mask, second_attention_mask):

        _, first_pooled_output = self.bert_model(first_input_ids, first_token_type_ids, first_attention_mask, output_all_encoded_layers=False)
        _, second_pooled_output = self.bert_model(second_input_ids, second_token_type_ids, second_attention_mask, output_all_encoded_layers=False)

        return first_pooled_output, second_pooled_output


class TextDataset(TensorDataset):
    def __init__(self, bert_tokenizer: BertTokenizer, jp_tokenizer:JumanTokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'dialogue_for_nsp' + '_cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples, \
                self.token_type_ids, \
                self.attention_mask, \
                self.next_sentence_label = pickle.load(handle)
        else:
            # キャッシュされたデータファイルがなければテキストファイルからデータセットを作成
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            # [CLS] A A A [SEP] B B B [SEP]
            self.token_type_ids = []
            #   0   0 0 0   0   1 1 1  1
            self.attention_mask = []
            #   1   1 1 1   1   1 1 1  1    0 0 0 0 ...
            self.next_sentence_label = []
            # [0, 1] 0: isNext, 1: notNext
            with open(file_path, encoding="utf-8") as f:
                docs = f.readlines()

            exsamples = []

            ZEN = "".join(chr(0xff01 + i) for i in range(94))
            HAN = "".join(chr(0x21 + i) for i in range(94))

            HAN2ZEN = str.maketrans(HAN, ZEN)

            num_doc = len(docs)
            for idx, line in enumerate(docs):

                text = line.rstrip(os.linesep)

                if text == "":
                    continue
                try:
                    next_text = docs[idx+1].rstrip(os.linesep)
                except IndexError:
                    continue
                if next_text == "":
                    continue

                if random.random() > args.nsp_swap_ratio:
                    while True:
                        rand_idx = random.randrange(0, num_doc)
                        next_text = docs[rand_idx].rstrip(os.linesep)
                        if (not next_text == "") and (rand_idx != idx+1):
                            break
                    nsp_label = 1
                    # random sequence
                else:
                    nsp_label = 0
                    # continuation sequence
                # jumanエラー対策
                text = text.replace(' ', '　' )
                next_text = next_text.replace(' ', '　' )
                text = mojimoji.han_to_zen(text, kana=False, digit=True, ascii=True)
                next_text = mojimoji.han_to_zen(next_text, kana=False, digit=True, ascii=True)
                text = text.translate(HAN2ZEN)
                next_text = next_text.translate(HAN2ZEN)
                # 元テキストを区切った状態に

                if len(text.encode('utf-8')) > 4096 or len(next_text.encode('utf-8')) > 4096:
                    continue

                first_tokenized_text = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(" ".join(jp_tokenizer.tokenize(text))))
                second_tokenized_text = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(" ".join(jp_tokenizer.tokenize(next_text))))

                fst_len = len(first_tokenized_text)
                scd_len = len(second_tokenized_text)
                # for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
                #    self.examples.append(bert_tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
                # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.
                
                # add special tokens
                # A A A (B B B) ->  [CLS] A A A [SEP] (B B B [SEP])
                added_special = bert_tokenizer.build_inputs_with_special_tokens(token_ids_0=first_tokenized_text,
                                                                                token_ids_1=second_tokenized_text)
                # token type ids
                type_ids = [0] * (2 + fst_len)
                scd_type = [1] * (1 + scd_len)
                type_ids += scd_type

                attention_mask = [1] * len(added_special)

                # Zero-pad up to the sequence length.
                diff = block_size - len(added_special)
                if diff < 0:
                    added_special = added_special[:diff]
                    type_ids = type_ids[:diff]
                    attention_mask = attention_mask[:diff]
                else:
                    padding = [0] * (block_size - len(added_special))
                    padding_1 = [0] * (block_size - len(added_special))
                    padding_2 = [0] * (block_size - len(added_special))
                    added_special += padding
                    type_ids += padding_1 
                    attention_mask += padding_2
                    
                assert len(added_special) == block_size
                assert len(type_ids) == block_size
                assert len(attention_mask) == block_size

                self.examples.append(added_special)
                self.token_type_ids.append(type_ids)
                self.attention_mask.append(attention_mask)
                self.next_sentence_label.append(nsp_label)
                
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump([self.examples, self.token_type_ids, self.attention_mask, self.next_sentence_label], handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # return torch.tensor(self.examples[item])
        return [torch.tensor(e) for e in [self.examples[item], self.token_type_ids[item], self.attention_mask[item], self.next_sentence_label[item]] ]


class BertSepInputNSPModel(BertPreTrainedModel):
    """BERT model for Next Sentence Prediction classification."""
    def __init__(self, config):
        super(BertSepInputNSPModel, self).__init__(config)
        self.num_labels = 2
        # isNext, notNext
        self.bert_for_double = MultiInputWrapper(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(1536, 2)
        self.apply(self.init_bert_weights)

    def forward(self, first_input_ids, second_input_ids,
            first_token_type_ids=None, second_token_type_ids=None,
            first_attention_mask=None, second_attention_mask=None, labels=None):

        # get encoded [CLS] tokens
        first_pooled_output, second_pooled_output = self.bert_for_double(first_input_ids, second_input_ids,
                                                                         first_token_type_ids, second_token_type_ids,
                                                                         first_attention_mask, second_attention_mask)
        first_dropped_output = self.dropout(first_pooled_output)
        second_dropped_output = self.dropout(second_pooled_output)
        union_output = torch.cat((first_dropped_output, second_dropped_output), 1)
        logits = self.classifier(union_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

# BertForPreTraining(BertPreTrainedModel)
# class BertConcatInputNSPModel(BertPreTrainedModel):
#     """BERT model for Next Sentence Prediction classification."""
#     def __init__(self, config):
#         super(BertConcatInputNSPModel, self).__init__(config)
#         self.num_labels = 2
#         # isNext, notNext
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(768, 2)
#         self.apply(self.init_bert_weights)
# 
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
# 
#         # get encoded [CLS] token
#         _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
#         dropped_output = self.dropout(pooled_output)
#         logits = self.classifier(union_output)
# 
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         else:
#             return logits

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def load_and_cache_examples(args, bert_tokenizer: BertTokenizer, jp_tokenizer: JumanTokenizer, evaluate=False):
    dataset = TextDataset(bert_tokenizer, jp_tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset


def train(args, train_dataset, model, bert_tokenizer: BertTokenizer, jp_tokenizer: JumanTokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * 1)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(bert_tokenizer))

    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if args.method == "method1" or args.method == "method3":
                # input: ids([CLS] A1 A2 A3 A4 [SEP] B1 B2 B3 B4 [SEP])
                # label: isNext or notNext
                inputs, token_type_ids, attention_mask, labels = batch
                inputs = inputs.to(args.device)
                token_type_ids = token_type_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
            else:
                # input1: ids([CLS] A1_1 A1_2 [SEP] A2_1 A2_2 [SEP])
                # input2: ids([CLS] B1_1 B1_2 [SEP] B2_1 B2_2 [SEP])
                # label: isNext or notNext
                inputs1, inputs2, labels = batch
                inputs1 = inputs1.to(args.device)
                inputs2 = inputs2.to(args.device)
                labels = labels.to(args.device)
            model.train()
            if args.method == "method1" or args.method == "method3":
                outputs = model(input_ids=inputs, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, next_sentence_label=labels)
            else:
                outputs = model(inputs1, inputs2, labels=labels)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            # 蓄積した勾配を何ステップで最適化に利用するか
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, bert_tokenizer, jp_tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, bert_tokenizer: BertTokenizer, jp_tokenizer: JumanTokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, bert_tokenizer, jp_tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        if args.method == "method1" or args.method == "method3":
            # input: ids([CLS] A1 A2 A3 A4 [SEP] B1 B2 B3 B4 [SEP])
            # label: isNext or notNext
            inputs, token_type_ids, attention_mask, labels = batch
            inputs = inputs.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
        else:
            # input1: ids([CLS] A1_1 A1_2 [SEP] A2_1 A2_2 [SEP])
            # input2: ids([CLS] B1_1 B1_2 [SEP] B2_1 B2_2 [SEP])
            # label: isNext or notNext
            inputs1, inputs2, labels = batch
            inputs1 = inputs1.to(args.device)
            inputs2 = inputs2.to(args.device)
            labels = labels.to(args.device)

        with torch.no_grad():
            if args.method == "method1" or args.method == "method3":
                outputs = model(input_ids=inputs, attention_mask=attention_mask,
                                token_type_ids=token_type_ids, next_sentence_label=labels)
            else:
                outputs = model(inputs1, inputs2, labels=labels)

            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    # ====== 学習 ======
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    # バリデーション
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    # ====== 学習オプション ======
    
    parser.add_argument("--method", default="method1", type=str,
                        help="NSP method.")
    parser.add_argument("--nsp_swap_ratio", default=0.5, type=float,
                        help="random Swap ratio of next sntences.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    # 重み減衰
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()


    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    device = torch.device('cuda:0')
    args.n_gpu = torch.cuda.device_count()
    # args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    -1, device, args.n_gpu, bool(False), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer

    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    bert_tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
    jp_tokenizer = JumanTokenizer()

    if args.block_size <= 0:
        args.block_size = bert_tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, bert_tokenizer.max_len_single_sentence)

    if args.method == "method1" or args.method == "method3":
        model = BertForNextSentencePrediction.from_pretrained(args.model_name_or_path,
                                                              from_tf=bool('.ckpt' in args.model_name_or_path),
                                                              config=config,
                                                              cache_dir=args.cache_dir if args.cache_dir else None)
         
        # ====== BERT一部パラメータ凍結 =======
        # - BERTエンコーダ最終層,プーラーのみ凍結回避
        bert_last_layer = copy.deepcopy(model.bert.encoder.layer[-1])
        bert_pooler = copy.deepcopy(model.bert.pooler)
        # - BERT凍結
        for param in model.bert.parameters():
            param.requires_grad = False
        # - 非凍結レイヤーで置換 
        model.bert.encoder.layer[-1] = bert_last_layer
        model.bert.pooler = bert_pooler
        # =====================================

    else:
        # 未完成
        model = BertSepInputNSPModel.from_pretrained(args.model_name_or_path,
                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
                                                    config=config,
                                                    cache_dir=args.cache_dir if args.cache_dir else None)
        # ====== BERT一部パラメータ凍結 =======
        # - BERTエンコーダ最終層,プーラーのみ凍結回避
        bert_last_layer = copy.deepcopy(model.bert_for_double.bert_model.encoder.layer[-1])
        bert_pooler = copy.deepcopy(model.bert_for_double.bert_model.pooler)
        # - BERT凍結
        for param in model.bert_for_double.bert_model.parameters():
            param.requires_grad = False
        # - 非凍結レイヤーで置換 
        model.bert_for_double.bert_model.encoder.layer[-1] = bert_last_layer
        model.bert_for_double.bert_model.pooler = bert_pooler
        # =====================================

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, bert_tokenizer, jp_tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, bert_tokenizer, jp_tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        bert_tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        bert_tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        
        if args.method == "method1" or args.method == "method3":
            model = BertForNextSentencePrediction.from_pretrained(args.model_name_or_path,
                                                                  from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                  config=config,
                                                                  cache_dir=args.cache_dir if args.cache_dir else None)
             
            # ====== BERT一部パラメータ凍結 =======
            # - BERTエンコーダ最終層,プーラーのみ凍結回避
            bert_last_layer = copy.deepcopy(model.bert.encoder.layer[-1])
            bert_pooler = copy.deepcopy(model.bert.pooler)
            # - BERT凍結
            for param in model.bert.parameters():
                param.requires_grad = False
            # - 非凍結レイヤーで置換 
            model.bert.encoder.layer[-1] = bert_last_layer
            model.bert.pooler = bert_pooler
            # =====================================

        else:
            # method2
            # 未完成
            model = BertSepInputNSPModel.from_pretrained(args.model_name_or_path,
                                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                                        config=config,
                                                        cache_dir=args.cache_dir if args.cache_dir else None)
            # ====== BERT一部パラメータ凍結 =======
            # - BERTエンコーダ最終層,プーラーのみ凍結回避
            bert_last_layer = copy.deepcopy(model.bert_for_double.bert_model.encoder.layer[-1])
            bert_pooler = copy.deepcopy(model.bert_for_double.bert_model.pooler)
            # - BERT凍結
            for param in model.bert_for_double.bert_model.parameters():
                param.requires_grad = False
            # - 非凍結レイヤーで置換 
            model.bert_for_double.bert_model.encoder.layer[-1] = bert_last_layer
            model.bert_for_double.bert_model.pooler = bert_pooler
            # =====================================

        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            if args.method == "method1" or args.method == "method3":
                model = BertForNextSentencePrediction.from_pretrained(args.model_name_or_path,
                                                                      from_tf=bool('.ckpt' in args.model_name_or_path),
                                                                      config=config,
                                                                      cache_dir=args.cache_dir if args.cache_dir else None)
                 
                # ====== BERT一部パラメータ凍結 =======
                # - BERTエンコーダ最終層,プーラーのみ凍結回避
                bert_last_layer = copy.deepcopy(model.bert.encoder.layer[-1])
                bert_pooler = copy.deepcopy(model.bert.pooler)
                # - BERT凍結
                for param in model.bert.parameters():
                    param.requires_grad = False
                # - 非凍結レイヤーで置換 
                model.bert.encoder.layer[-1] = bert_last_layer
                model.bert.pooler = bert_pooler
                # =====================================

            else:
                # method2
                # 未完成
                model = BertSepInputNSPModel.from_pretrained(args.model_name_or_path,
                                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                                            config=config,
                                                            cache_dir=args.cache_dir if args.cache_dir else None)
                # ====== BERT一部パラメータ凍結 =======
                # - BERTエンコーダ最終層,プーラーのみ凍結回避
                bert_last_layer = copy.deepcopy(model.bert_for_double.bert_model.encoder.layer[-1])
                bert_pooler = copy.deepcopy(model.bert_for_double.bert_model.pooler)
                # - BERT凍結
                for param in model.bert_for_double.bert_model.parameters():
                    param.requires_grad = False
                # - 非凍結レイヤーで置換 
                model.bert_for_double.bert_model.encoder.layer[-1] = bert_last_layer
                model.bert_for_double.bert_model.pooler = bert_pooler
                # =====================================

            model.to(args.device)
            result = evaluate(args, model, bert_tokenizer, jp_tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
