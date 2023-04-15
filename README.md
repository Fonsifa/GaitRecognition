# 基于手机惯性信号的步态识别方法
## Authentication
**步态身份认证**
models.py 包括一些对比方法
ConvGTN则是利用一维卷积改进的GTN方法
auth.py 用于训练和测试

## Identification
**步态身份识别**
CNN+Transformer为我们提出的模型

GTN骨架来源于
>https://github.com/ZZUFaceBookDL/GTN

Self-Attention部分改自
>https://github.com/ghsama/ConvTransformerTimeSeries/blob/master/net.py

身份认证Transformer骨架改自
>https://github.com/gzerveas/mvts_transformer/blob/master/src/models/ts_transformer.py