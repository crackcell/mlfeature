package org.apache.spark.ml.feature

import org.apache.spark.MyTestSuite
import org.apache.spark.sql.Row

/**
  * Created by Menglong TAN on 3/22/17.
  */
class DataBalancerSuite extends MyTestSuite {

  import sqlContext.implicits._

  test("Build ratio map") {
    val data: Seq[String] = Seq("a", "a", "a", "a", "b","b", "c")
    val expectedFactor = Map("a" -> 1.0, "b" -> 2.0, "c" -> 4.0)
    val expectedNum = Map("a" -> 4, "b" -> 4, "c" -> 4)
    val dataFrame = data.toDF("feature")

    val balancer = new DataBalancer()
      .setInputCol("feature")
      .setOutputCol("result")

    val model = balancer.fit(dataFrame)

    model.factors.foreach { case (value, ratio) =>
      assert(ratio === expectedFactor(value))
    }

    model.transform(dataFrame).groupBy("feature").count().collect().map {
      case Row(feature: String, count) =>
        assert(count === expectedNum(feature))
    }
  }
}
