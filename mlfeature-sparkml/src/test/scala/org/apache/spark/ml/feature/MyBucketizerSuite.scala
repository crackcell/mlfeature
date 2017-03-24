package org.apache.spark.ml.feature

import org.apache.spark.{MySparkTestSuite, SparkException}
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Created by crackcell on 3/13/17.
  */
class MyBucketizerSuite extends MySparkTestSuite {

  import sqlContext.implicits._

  test("Bucket continuous features, without -inf,inf") {
    // Check a set of valid feature values.
    val splits = Array(-0.5, 0.0, 0.5)
    val validData = Array(-0.5, -0.3, 0.0, 0.2)
    val expectedBuckets = Array(0.0, 0.0, 1.0, 1.0)
    val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

    val bucketizer: MyBucketizer = new MyBucketizer()
      .setInputCol("feature")
      .setOutputCol("result")
      .setSplits(splits)

    bucketizer.transform(dataFrame).select("result", "expected").collect().foreach {
      case Row(x: Double, y: Double) =>
        assert(x === y,
          s"The feature value is not correct after bucketing.  Expected $y but found $x")
    }

    // Check for exceptions when using a set of invalid feature values.
    val invalidData1: Array[Double] = Array(-0.9) ++ validData
    val invalidData2 = Array(0.51) ++ validData
    val badDF1 = invalidData1.zipWithIndex.toSeq.toDF("feature", "idx")
    withClue("Invalid feature value -0.9 was not caught as an invalid feature!") {
      intercept[SparkException] {
        bucketizer.transform(badDF1).collect()
      }
    }
    val badDF2 = invalidData2.zipWithIndex.toSeq.toDF("feature", "idx")
    withClue("Invalid feature value 0.51 was not caught as an invalid feature!") {
      intercept[SparkException] {
        bucketizer.transform(badDF2).collect()
      }
    }
  }

  test("Bucket continuous features, with -inf,inf") {
    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
    val validData = Array(-0.9, -0.5, -0.3, 0.0, 0.2, 0.5, 0.9)
    val expectedBuckets = Array(0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
    val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

    val bucketizer: MyBucketizer = new MyBucketizer()
      .setInputCol("feature")
      .setOutputCol("result")
      .setSplits(splits)

    bucketizer.transform(dataFrame).select("result", "expected").collect().foreach {
      case Row(x: Double, y: Double) =>
        assert(x === y,
          s"The feature value is not correct after bucketing.  Expected $y but found $x")
    }
  }

  test("Bucket continuous features, with NaN data but non-NaN splits") {
    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
    val validData = Array(-0.9, -0.5, -0.3, 0.0, 0.2, 0.5, 0.9, Double.NaN, Double.NaN, Double.NaN)
    val expectedBuckets = Array(0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0)
    val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

    val bucketizer: MyBucketizer = new MyBucketizer()
      .setInputCol("feature")
      .setOutputCol("result")
      .setSplits(splits)

    bucketizer.setHandleInvalid("keep")
    bucketizer.transform(dataFrame).select("result", "expected").collect().foreach {
      case Row(x: Double, y: Double) =>
        assert(x === y,
          s"The feature value is not correct after bucketing.  Expected $y but found $x")
    }

    bucketizer.setHandleInvalid("skip")
    val skipResults: Array[Double] = bucketizer.transform(dataFrame)
      .select("result").as[Double].collect()
    assert(skipResults.length === 7)
    assert(skipResults.forall(_ !== 4.0))

    bucketizer.setHandleInvalid("error")
    withClue("Bucketizer should throw error when setHandleInvalid=error and given NaN values") {
      intercept[SparkException] {
        bucketizer.transform(dataFrame).collect()
      }
    }
  }

  test("Bucket continuous features, with NaN splits") {
    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity, Double.NaN)
    withClue("Invalid NaN split was not caught during Bucketizer initialization") {
      intercept[IllegalArgumentException] {
        new Bucketizer().setSplits(splits)
      }
    }
  }

  test("Bucket continuous features, with values out of bounds and NegativeInfinity lower bound") {
    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5)
    val validData = Array(-0.9, -0.5, -0.3, 0.0, 0.2, 0.5, 0.9, Double.NaN, Double.NaN, Double.NaN)
    val expectedBuckets = Array(0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0)
    val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

    val bucketizer: MyBucketizer = new MyBucketizer()
      .setInputCol("feature")
      .setOutputCol("result")
      .setSplits(splits)

    bucketizer.setHandleInvalid("keep")
    bucketizer.transform(dataFrame).select("result", "expected").collect().foreach {
      case Row(x: Double, y: Double) =>
        assert(x === y,
          s"The feature value is not correct after bucketing.  Expected $y but found $x")
    }

    bucketizer.setHandleInvalid("skip")
    val skipResults: Array[Double] = bucketizer.transform(dataFrame)
      .select("result").as[Double].collect()
    assert(skipResults.length === 6)
    assert(skipResults.forall(_ !== 4.0))

    bucketizer.setHandleInvalid("error")
    withClue("Bucketizer should throw error when setHandleInvalid=error and given NaN values") {
      intercept[SparkException] {
        bucketizer.transform(dataFrame).collect()
      }
    }
  }

  test("Bucket continuous features, with values out of bounds and PositiveInfinity lower bound") {
    val splits = Array(-0.5, 0.0, 0.5, Double.PositiveInfinity)
    val validData = Array(-0.9, -0.5, -0.3, 0.0, 0.2, 0.5, 0.9, Double.NaN, Double.NaN, Double.NaN)
    val expectedBuckets = Array(3.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0)
    val dataFrame: DataFrame = validData.zip(expectedBuckets).toSeq.toDF("feature", "expected")

    val bucketizer: MyBucketizer = new MyBucketizer()
      .setInputCol("feature")
      .setOutputCol("result")
      .setSplits(splits)

    bucketizer.setHandleInvalid("keep")
    bucketizer.transform(dataFrame).select("result", "expected").collect().foreach {
      case Row(x: Double, y: Double) =>
        assert(x === y,
          s"The feature value is not correct after bucketing.  Expected $y but found $x")
    }

    bucketizer.setHandleInvalid("skip")
    val skipResults: Array[Double] = bucketizer.transform(dataFrame)
      .select("result").as[Double].collect()
    assert(skipResults.length === 5)
    assert(skipResults.forall(_ !== 4.0))

    bucketizer.setHandleInvalid("error")
    withClue("Bucketizer should throw error when setHandleInvalid=error and given NaN values") {
      intercept[SparkException] {
        bucketizer.transform(dataFrame).collect()
      }
    }
  }

}
