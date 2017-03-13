# spark-ml-fea
Spark MLlib特征工程工具包：

## 数据抽样

- DataBalancer：通过随机oversample制备平衡样本

## 缺失值处理

- MissingValueMeanImputer：用均值填充缺失数据

## 特征转换

### MyBucketizer：官方Bucketizer的增强版本

增加了对NULL值和越界数据的处理。和NaN一样，NULL和越界数据会被放到一个特殊的分桶中

示例： 
```scala

```
### MyStringIndxer：官方StringIndxer的增强版本：

给handleInvalid增加了keep选项，对于NULL值和没有见过的label，会被索引到最后一个下标里面

