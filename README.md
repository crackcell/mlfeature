# MLfeature

Feature engineering toolkit for Spark MLlib:
- Data preprocessing:
  - Handle imbalanced dataset: DataBalancer
  - ~~Handle missing values~~: (Implemented in Spark 2.2, [SPARK-13568](https://github.com/apache/spark/pull/11601))
    - ~~Impute continuous missing values with mean: MissingValueMeanImputor~~
- Feature selection:
  - VarianceSelector: remove fetures with low variance
  - ByModelSelector: select feature with model
- Feature transformers:
  - Enhanced Bucketizer: MyBucketizer (Waiting to be merged, [SPARK-19781](https://github.com/apache/spark/pull/17123))
  - ~~Enhanced StringIndexer: MyStringIndexer~~ (Merged with Spark 2.2, [SPARK-17233](https://github.com/apache/spark/pull/17233))

## Handle imbalcned dataset

- DataBalancer: Make an balanced dataset with multiple strategies:
  - Re-sampling:
    - over-sampling
    - under-sampling
    - middle-sampling
  - SMOTE: TODO
  
Example:

```scala
val data = Array("a", "a", "b", "c")
val dataFrame = data.toSeq.toDF("feature")

val balancer = new DataBalancer()
  .setStrategy("oversampling")
  .setInputCol("feature")

val result = balacner.transform(dataFrame)
result.show(100)
```

```scala
val data: Seq[String] = Seq("a", "a", "a", "a", "b", "b","b", "c")
val dataFrame = data.toDF("feature")

val balancer = new DataBalancer()
  .setStrategy("undersampling")
  .setInputCol("feature")

val result = balancer.transform(dataFrame)
result.show(100)
```

```scala
val data: Seq[String] = Seq("a", "a", "a", "a", "b", "b","b", "c")
val dataFrame = data.toDF("feature")

val balancer = new DataBalancer()
  .setStrategy("middlesampling")
  .setInputCol("feature")

val result = balancer.transform(dataFrame)
result.show(100)
```

## Handle missing values

- MissingValueMeanImputer: Impute continuous missing values with mean

# Feature Selection

## VarianceSelector

VarianceSelector is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

```scala
val data = Array(
  Vectors.dense(0, 1.0, 0),
  Vectors.dense(0, 3.0, 0),
  Vectors.dense(0, 4.0, 0),
  Vectors.dense(0, 5.0, 0),
  Vectors.dense(1, 6.0, 0)
)

val expected = Array(
  Vectors.dense(1.0),
  Vectors.dense(3.0),
  Vectors.dense(4.0),
  Vectors.dense(5.0),
  Vectors.dense(6.0)
)

val df = data.zip(expected).toSeq.toDF("features", "expected")

val selector = new VarianceSelector()
  .setInputCol("features")
  .setOutputCol("selected")
  .setThreshold(3)

val result = selector.transform(df)

result.select("expected", "selected").collect()
  .foreach { case Row(vector1: Vector, vector2: Vector) =>
    assert(vector1.equals(vector2), "Transformed vector is different with expected.")
  }
```

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

### MyStringIndxer: Enhanced StringIndexer

Give NULLs and unseen lables a special index.

Example:
```scala
val data = Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
val df = data.toDF("id", "label")
val indexer = new MyStringIndexer()
  .setInputCol("label")
  .setOutputCol("labelIndex")
  .fit(df)

val transformed = indexer.transform(df)
```

## ByModelSelector

