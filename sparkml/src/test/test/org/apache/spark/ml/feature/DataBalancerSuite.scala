package org.apache.spark.ml.feature

import org.apache.spark.MyTestSuite
import org.apache.spark.sql.Row

/**
  * Created by Menglong TAN on 3/22/17.
  */
class DataBalancerSuite extends MyTestSuite {

  test("Build ratio map") {
    val data: Seq[String] = Seq("a", "a", "a", "a", "b","b", "c")
    val expectedNum = Map("a" -> 4, "b" -> 4, "c" -> 4)
    val dataFrame = data.toDF("feature")

    val balancer = new DataBalancer()
      .setInputCol("feature")

    balancer.transform(dataFrame).groupBy("feature").count().collect().map {
      case Row(feature: String, count) =>
        assert(count === expectedNum(feature))
    }
  }

}
