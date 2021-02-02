# knowledge-aug-adv
This is PyTorch implementation of the Commonsense Knowledge Augmentation for low-resource languages via adversarial learning.

** Train **

Train mBERT: src En, tgt Kr

```
python train_mbert_clip.py \
    --target_lang kr \
    --src_data_dir data/en_triples \
    --tgt_data_dir data/kr_triples \
    --do_train \
    --do_eval \
    --learning_rate 2e-5 \
    --batch_size 64 \
    --n_critic 1 \
    --model_save_file ./save/adv/kr \
    --max_epoch 1 \
    --clip_lower -1 \
    --clip_upper 1 \
    --lambd 0.1 \
    --dev_file kr_dev.tsv \
    --test_file kr_test.tsv \
    --max_step 1000
```
