package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{Identifiable, MLWritable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by Menglong TAN on 3/21/17.
  */
private[feature] trait StratifiedSamplingDataBalancerBase
  extends Params with HasInputCol with HasOutputCol {

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains($(inputCol)), s"Input column ${$(inputCol)} does not exist")
    require(!schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exist")
    schema
  }

  protected def resample(datasets: Array[(DataFrame, Double)]): DataFrame = {
    var all = datasets(0)._1
    for (i <- 1 to datasets.length - 1) all = {
      val dataset = datasets(i)._1
      val ratio = datasets(i)._2
      all.unionAll(dataset.sample(false, ratio))
    }
    all
  }

}

class StratifiedSamplingDataBalancer(override val uid: String)
  extends Estimator[StratifiedSamplingDataBalancerModel] with StratifiedSamplingDataBalancerBase {

  def this(ratio: Map[AnyVal, Double]) =
    this(Identifiable.randomUID("stratifiedSamplingDataBalancer"), ratio)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): StratifiedSamplingDataBalancerModel = {
    validateAndTransformSchema(dataset.schema)
  }

  override def transformSchema(schema: StructType) = {
    validateAndTransformSchema(schema)
  }
}

class StratifiedSamplingDataBalancerModel(
    override val uid: String,
    val ratio: Map[String, Double])
  extends Model[StratifiedSamplingDataBalancerModel] with StratifiedSamplingDataBalancerBase {

  def this(ratio: Map[AnyVal, Double]) =
    this(Identifiable.randomUID("stratifiedSamplingDataBalancer"), ratio)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val datasets = ArrayBuffer[(DataFrame, Double)]()

    ratio.foreach { case (v, r) =>
      datasets.append((dataset.filter(s"${$(inputCol)} = ${v}").toDF(), r))
    }

    resample(datasets.toArray)
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): StratifiedSamplingDataBalancerModel = {
    val copied = new StratifiedSamplingDataBalancerModel(uid, ratio)
    copyValues(copied, extra).setParent(parent)
  }
}

