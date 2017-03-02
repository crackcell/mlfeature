# spark-ml-fea
Spark MLlib特征工程工具包：

## 数据抽样

- DataBalancer：通过随机oversample制备平衡样本

## 缺失值处理

- MissingValueMeanImputer：用均值填充缺失数据

## 数据处理

- MyBucketizer：数据分桶器。官方Bucketizer的增强版本。增加了对NULL值的处理。和NaN一样，NULL会被放到一个特殊的分桶中
- MyStringIndxer：数据索引器。官方StringIndxer的增强版本。参考Bucketizer的风格，给handleInvalid增加了keep选项，对于NULL值，会被索引到最后一个下标里面
