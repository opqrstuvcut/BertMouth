# BertMouth
This repository is a reimplementation of the paper (BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model: https://arxiv.org/abs/1902.04094).
                                      
## Requirement
- Python3
- PyTorch (1.0+)
- Transformers (2.1.1)
- NumPy
- tqdm 

A trained Pokemon text generation model is https://github.com/opqrstuvcut/pokemon_bert_model.
                                                                                                                                                                      
## Usage                                                                                                                                                                                                   
### generate text

Command example:
```
python ./bert_mouth.py \
--bert_model ./pokemon_model \
--do_generate \
--seq_length 20 \
--max_iter 5
```

- bert_model: Trained model path.                                                                                                                                                                     
- max_seq_length: Maximum sequence length in BERT. This value depends on pretraining setting. 
- seq_length: Generated text length.
- max_iter: Number of iteration in the Gibbs sampling.

generated text examples:
1. 時速は３００キロのスピード。真っ暗なヒレを使い戦うが数は少ない。
2. 気づかずかみちぎる雷のような雷雲は強烈なキックを秘める武器だ。
3. 戦いでは溶けてしまうがその皮膚は鋭い渦巻き。戦うときにとても便利。 
4. 優しい心の持ち主。生まれたときブーバーに鍛えられた１本は暗闇でも見えるぞ。
5. 自分がポケモンになったときこの姿に変わるのだ。強烈なキックは世界一ムチおよぼす威力。
6. １匹では生きられないように群れで行動し１匹で生命力が強くなった。体は硬い。
7. 頭に生える真っ白は大人１００人以上。チビノーズのような風に乗って大きな岩を砕く。
8. メタモンの頭のしっぽだけが大好きでとても大好きだがそれをかじって見かけるだけだ。
9. なんでも食べる。両腕の鋭いカマで相手を包みこみ怒らせたところに巻きつき次の日には風に乗って空を飛ぶ。
10. 明かりのカマは１トン。暗闇でもかみちぎるシッポのようなトゲは強い脚力でふくらむ。

### train on your data 

Command example:
```
  python ./bert_mouth.py \
  --bert_model ./Japanese_L-12_H-768_A-12_E-30_BPE/ \                                                                                                                                               
  --output_dir ./models \
  --train_file train.txt \
  --valid_file valid.txt \
  --max_seq_length 128 \
  --do_train \
  --train_batch_size 10 \
  --num_train_epochs 100
  ```

- bert_model: Pretrained BERT model path.
- output_dir: Save path.
- train_file: Training file path.
- valid_file: Validation file path.
- max_seq_length: Maximum sequence length in BERT. This value depends on pretraining setting. 
- train_batch_size: Batch size.
- num_train_epochs Number of epochs.

The format of training data and validation data: 
```
token_A,1 tokenA,2 ... tokenA,An
token_B,1 tokenB,2 ... tokenB,Bn
︙
```
Each row is tokenized by a tokenizer which is used in pretraining.

Training data example:
```
生まれた とき から 背中 に 不思議な タネ が 植えて あって 体 と ともに 育つ と いう 。
養分 を 摂って 大きく なった つぼみ から 香り が 漂い だす と もう すぐ ハナ が 開く 証拠 だ 。
生まれた とき から 尻尾 に 炎 が 点 って いる 。 炎 が 消えた とき その 命 は 終わって しまう 。
```

