# MLfeature

Feature engineering toolkit for Spark MLlib:
- Data preprocessing:
  - Handle imbalanced dataset: DataBalancer
  - Handle missing values:
    - Impute continuous missing values with mean: MissingValueMeanImputor
    - Impute continuous missing values with conditional mean: TODO
- Feature transform
  - Enhanced Bucketizer: MyBucketizer
  - ~~Enhanced StringIndexer: MyStringIndexer~~ (Merged with Spark master, [SPARK-17233](https://github.com/apache/spark/pull/17233))
- Feature selection
  - AuROC for single feature: TODO
  - Correlation: TODO

## Handle imbalcned dataset

- DataBalancer: Oversample balancer

## Handle missing values

- MissingValueMeanImputer: Impute continuous missing values with mean

## Feature transform

### MyBucketizer: Enhanced Bucketizer

Put NULLs and values out of bounds into a special bucket as well as NaN.

Example:
```scala
val splits = Array(-0.5, 0.0, 0.5)
val validData = Array(-0.5, -0.3, 0.0, 0.2)
val expectedBuckets = Array(0.0, 0.0, 1.0, 1.0)
val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

val bucketizer: MyBucketizer = new MyBucketizer()
  .setInputCol("feature")
  .setOutputCol("result")
  .setSplits(splits)

val transformed = bucketizer.transform(dataFrame)
```
