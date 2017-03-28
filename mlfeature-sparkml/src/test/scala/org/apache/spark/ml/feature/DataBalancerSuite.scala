package org.apache.spark.ml.feature

import org.apache.spark.MySparkTestSuite
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

/**
  * Created by Menglong TAN on 3/22/17.
  */
class DataBalancerSuite extends MySparkTestSuite {

  import sqlContext.implicits._

  test("Over-sampling") {
    val data: Seq[String] = Seq("a", "a", "a", "a", "b", "b", "c")
    val expectedNum = Map("a" -> 4, "b" -> 4, "c" -> 4)
    val dataFrame = data.toDF("feature")

    val balancer = new DataBalancer()
      .setStrategy("oversampling")
      .setInputCol("feature")

    val result = balancer.transform(dataFrame)
  }

  test("Under-sampling") {
    val data: Seq[String] = Seq("a", "a", "a", "a", "b", "b","b", "c")
    val expectedNum = Map("a" -> 1, "b" -> 1, "c" -> 1)
    val dataFrame = data.toDF("feature")

    val balancer = new DataBalancer()
      .setStrategy("undersampling")
      .setInputCol("feature")

    val result = balancer.transform(dataFrame)
  }

  test("Middle-sampling") {
    val data: Seq[String] = Seq("a", "a", "a", "a", "b", "b","b", "c")
    val expectedNum = Map("a" -> 1, "b" -> 1, "c" -> 1)
    val dataFrame = data.toDF("feature")

    val balancer = new DataBalancer()
      .setStrategy("middlesampling")
      .setInputCol("feature")

    val result = balancer.transform(dataFrame)
  }

}
