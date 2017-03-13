# spark-ml-fea

Spark MLlib特征工程工具包，包含了一些实际工作中常用功能：
- 数据预处理：
  - 制备平衡样本：DataBalancer
  - 处理缺失值：MissingValueMeanImputor
- 特征转换
  - 增强版的连续值分桶器：MyBucketizer
  - 增强版的字符串转离散值的索引器：MyStringIndexer

## 不平衡样本

- DataBalancer：通过随机oversample制备平衡样本

## 缺失值处理

- MissingValueMeanImputer：用均值填充缺失数据

## 特征转换

### MyBucketizer：官方Bucketizer的增强版本

增加了对NULL值和越界数据的处理。和NaN一样，NULL和越界数据会被放到一个特殊的分桶中

示例： 
```scala
val splits = Array(-0.5, 0.0, 0.5)
val validData = Array(-0.5, -0.3, 0.0, 0.2)
val expectedBuckets = Array(0.0, 0.0, 1.0, 1.0)
val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

val bucketizer: MyBucketizer = new MyBucketizer()
  .setInputCol("feature")
  .setOutputCol("result")
  .setSplits(splits)

bucketizer.transform(dataFrame)
```
### MyStringIndxer：官方StringIndxer的增强版本：

给handleInvalid增加了keep选项，对于NULL值和没有见过的label，会被索引到最后一个下标里面

示例：
```scala
val data = Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
val df = data.toDF("id", "label")
val indexer = new MyStringIndexer()
  .setInputCol("label")
  .setOutputCol("labelIndex")
  .fit(df)

val transformed = indexer.transform(df)
```