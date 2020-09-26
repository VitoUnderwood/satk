# 论文阅读笔记

## 任务介绍

针对细粒度的情感挖掘任务，相关工作研究只是单独的提取aspect或者opinion，往往忽略了二者之间特定的联系，但是这种联系对于一些下游任务来说确实至关重要的， 例如情感分类任务，摘要任务， 所以提出AOPE aspect opinion 对抽取任务

## Synchronous Double-channel Recurrent Network (SDRN) 模型

同时抽取opinion entity和relation

### base bert

使用基本bert作为句子的编码器，获得句子的context representation

### opinion entity extraction unit

channel 1 抽取aspect opinion， 实际上是一个 语句标注任务， 采用CRF条件随机场

### relation detection unit

channel 2 抽取aspect和opinion之间的relation

### synchroniza- tion unit

### inference layer

Entity Synchronization Mechanism (ESM) and Relation Synchronization Mechanism (RSM) to 
enhance the mutual benefit on the above two channels.

## 训练结果

```bash
Epoch: 69/70
0.00030362565580828606
     Instance: 0; Time: 0.11s; loss: 0.3528; target_acc: 134/134=1.0000; relation_acc: 12/12=1.0000
     Instance: 200; Time: 6.20s; loss: 7.2610; target_acc: 4262/4262=1.0000; relation_acc: 557/558=0.9982
     Instance: 400; Time: 5.76s; loss: 7.8888; target_acc: 8248/8250=0.9998; relation_acc: 963/970=0.9928
     Instance: 600; Time: 6.08s; loss: 7.3071; target_acc: 12262/12265=0.9998; relation_acc: 1288/1296=0.9938
     Instance: 800; Time: 6.04s; loss: 6.7854; target_acc: 16333/16336=0.9998; relation_acc: 1602/1610=0.9950
     Instance: 1000; Time: 6.28s; loss: 7.2507; target_acc: 20164/20167=0.9999; relation_acc: 2089/2102=0.9938
     Instance: 1200; Time: 6.07s; loss: 7.5737; target_acc: 24336/24341=0.9998; relation_acc: 2499/2528=0.9885
     Instance: 1400; Time: 6.10s; loss: 7.2108; target_acc: 28274/28280=0.9998; relation_acc: 2888/2918=0.9897
     Instance: 1600; Time: 6.46s; loss: 7.1735; target_acc: 32381/32387=0.9998; relation_acc: 3438/3470=0.9908
     Instance: 1800; Time: 6.30s; loss: 8.3658; target_acc: 36429/36439=0.9997; relation_acc: 3919/3954=0.9911
     Instance: 2000; Time: 6.26s; loss: 6.7777; target_acc: 40301/40312=0.9997; relation_acc: 4196/4232=0.9915
     Instance: 2200; Time: 6.20s; loss: 7.1103; target_acc: 44487/44498=0.9998; relation_acc: 4632/4668=0.9923
     Instance: 2400; Time: 5.65s; loss: 7.0776; target_acc: 48365/48376=0.9998; relation_acc: 4976/5012=0.9928
     Instance: 2600; Time: 5.73s; loss: 7.8197; target_acc: 52198/52211=0.9998; relation_acc: 5368/5404=0.9933
     Instance: 2800; Time: 6.10s; loss: 7.0808; target_acc: 56324/56337=0.9998; relation_acc: 5846/5882=0.9939
     Instance: 3000; Time: 6.34s; loss: 7.8981; target_acc: 60530/60547=0.9997; relation_acc: 6280/6318=0.9940
     Instance: 3040; Time: 1.41s; loss: 1.6093; target_acc: 61310/61327=0.9997; relation_acc: 6412/6450=0.9941
Epoch: 69 training finished. Time: 93.09s, speed: 32.71st/s,  total loss: 112.54307164981583
totalloss: 112.54307164981583
test: time: 10.46s, speed: 0.00st/s

▽
                        step * args.batchSize, temp_cost, sample_loss, right_target_token, whole_target_token,
relation result: Precision: 0.5904; Recall: 0.6789; F1: 0.6316
target result: Precision: 0.8473; Recall: 0.7902; F1: 0.8177
opinion result: Precision: 0.7448; Recall: 0.8538; F1: 0.7956
```

## 测试结果

```bash
test: time: 10.85s, speed: 0.00st/s
relation result: Precision: 0.6481; Recall: 0.7368; F1: 0.6897
target result: Precision: 0.8535; Recall: 0.8208; F1: 0.8368
opinion result: Precision: 0.7792; Recall: 0.8863; F1: 0.8293
```