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

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{DoubleParam, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.types.StructType


/**
  * Class description.
  *
  * @author Menglong TAN
  */
trait VarianceThresholdSelectorParams
  extends Params with HasInputCol with HasOutputCol with DefaultParamsWritable {

  val threshold: DoubleParam = new DoubleParam(this, "threshold",
    "lower bound of the feature variance")

  def getVariance: Double = $(threshold)
}

class VarianceThresholdSelector(override val uid: String)
  extends Transformer with VarianceThresholdSelectorParams {

  def this() = this(Identifiable.randomUID("varThres"))

  setDefault(threshold -> 0)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setVariance(value: Double): this.type = set(threshold, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val input: RDD[OldVector] = dataset.select($(inputCol)).rdd.map {
      case Row(v: Vector) => OldVectors.fromML(v)
    }
    val selectedIndics = Statistics.colStats(input).variance.toArray
      .zipWithIndex
      .filter(_._1 >= $(threshold))
      .map(_._2)

    val slicer = new VectorSlicer()
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
      .setIndices(selectedIndics)

    slicer.transform(dataset)
  }

  override def transformSchema(schema: StructType): StructType = {
    val dataType = new VectorUDT
    require(isDefined(inputCol),
      s"VarianceSelector requires input column parameter: ${$(inputCol)}")
    require(!schema.fieldNames.contains($(outputCol)),
      s"Output column already exists: ${$(outputCol)}")
    SchemaUtils.checkColumnType(schema, $(inputCol), dataType)
    SchemaUtils.appendColumn(schema, $(outputCol), dataType)
  }

  override def copy(extra: ParamMap): VarianceThresholdSelector = defaultCopy(extra)
}

object VarianceThresholdSelector extends DefaultParamsReadable[VarianceThresholdSelector] {
  override def load(path: String): VarianceThresholdSelector = super.load(path)
}
