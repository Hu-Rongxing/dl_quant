本程序，主要运用darts包，进行股票量化交易。用到的模型主要有：LightGBMModel、TSMixerModel、TFTModel、NHiTSModel、LSTM模型，以及上述模型的集成模型RegressionEnsembleModel。# 包的运行

# 包的结构

## data 保存程序运行所需的或生成的数据


## execution 通过apscheduler包按计划调度程序。

## load_data 生成程序运行所需数据。

## logs 保存程序运行日志。


## models 预测股票的模型。

## notebooks 研究用的notebook。

## qmt 操作迅投客户端。

## tests 测试模块。

## utils 各类工具。

### logger 模块

logger提供以下级别的日志：

- logger.debug
- logger.info
- logger.warning
- logger.trader: 用于记录交易日志，该级别及其以上级别的日志邮件通知作者。
- logger.error
- logger.fetal

## xtquant 量化程序接口，由迅投官方提供，不需要更改。

# 参考资料

1. [darts官方网站：https://unit8co.github.io/darts/README.html](https://https://unit8co.github.io/darts/README.html)
