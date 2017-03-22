package org.apache.spark.ml.feature

import org.apache.spark.MyTestSuite

/**
  * Created by Menglong TAN on 3/22/17.
  */
class DataBalancerSuite extends MyTestSuite {

  import sqlContext.implicits._

  test("Build ratio map") {
    val data = Array("a", "a", "b", "c")
    val expectedRatio: Map[Any, Double] = Map("a" -> 1.0, "b" -> 2.0, "c" -> 2.0)
    val dataFrame = data.toSeq.toDF("feature")

    val balancer = new DataBalancer()
      .setInputCol("feature")
      .setOutputCol("result")

    val model = balancer.fit(dataFrame)

    model.factors.foreach { case (value, ratio) =>
      assert(ratio === expectedRatio(value))
    }

    model.transform(dataFrame).show(100)
  }
}
