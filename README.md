# 汽车行业用户观点主题及情感识别 baseline 63+  
## 基本思路：  
一个 content 可能对应多个主题，每个主题有对应的情感值，10*3 分三十个标签，标签构建做法如下  
```
train_df['label'] = train_df['subject'].str.cat(train_df['sentiment_value'].astype(str))
```
1. 输入：模型使用 word level 可以考虑 char level embedding，因为有些词滥用情况导致分词结果在预训练词向量中找不到对应向量表示。
2. 输出：sigmoid(3*10)，因为 30 个位置有 1 个或多个为 1

**使用词向量**:  
[词向量][https://github.com/Embedding/Chinese-Word-Vectors]  
**capsule 实现参考（照搬）**:  
[capsule 实现][https://github.com/bojone/Capsule]  

ps. 数据处理、模型设置都还比较粗糙，欢迎大家给我提改进建议。
