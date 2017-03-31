/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark.ml.feature

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row

import org.apache.spark.MySparkTestSuite

/**
  * Class description.
  *
  * @author Menglong TAN
  */
class VarianceSelectorSuite extends MySparkTestSuite {

  import sqlContext.implicits._

  test("VarianceSelector remove features with low variance") {
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
  }
}
