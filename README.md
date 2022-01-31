# knowledge-aug-adv
PyTorch implementation of the [Commonsense Knowledge Augmentation for low-resource languages via adversarial learning](https://ojs.aaai.org/index.php/AAAI/article/view/16793).

## Train

Train mBERT: src En, tgt Kr

```
python train_mbert_clip.py \
    --target_lang kr \
    --src_data_dir data/en_triples \
    --tgt_data_dir data/kr_triples \
    --do_train \
    --do_eval \
    --learning_rate 1e-5 \
    --batch_size 64 \
    --n_critic 1 \
    --model_save_file ./save/adv/kr \
    --max_epoch 1 \
    --clip_lower -1 \
    --clip_upper 1 \
    --lambd 0.1 \
    --dev_file kr_dev.tsv \
    --test_file kr_test.tsv \
    --max_step 2000
```
## Test

```
python train_mbert_clip.py \
    --target_lang kr \
    --src_data_dir data/en_triples \
    --tgt_data_dir data/kr_triples \
    --do_eval \
    --batch_size 64 \
    --test_model {test_model_path} \
    --test_file kr_test.tsv
```

## Data Augmentation

```
python train_mbert_clip.py \
    --target_lang kr \
    --tgt_data_dir data/kr_triples \
    --do_aug \
    --test_model {model_path_for_aug} \
    --test_file kr_test.tsv
```