package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol, HasSeed, HasStrategy}
import org.apache.spark.ml.util.{Identifiable, MLWritable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by Menglong TAN on 3/21/17.
  */
private[feature] trait DataBalancerBase
  extends Params with HasInputCol with HasOutputCol with HasSeed {

  val strategy: Param[String] = new Param[String](this, "strategy",
    "how to handle imbalanced dataset. Options are oversampling",
    ParamValidators.inArray(DataBalancerBase.supportedStrategies))

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains($(inputCol)), s"Input column ${$(inputCol)} does not exist")
    require(!schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exist")
    schema
  }

}

private object DataBalancerBase {

  private[feature] val OVERSAMPLING_STRATEGY: String = "oversampling"
  private[feature] val supportedStrategies: Array[String] =
    Array(OVERSAMPLING_STRATEGY)

}

class DataBalancer(override val uid: String)
  extends Estimator[DataBalancerModel] with DataBalancerBase {

  def this() =
    this(Identifiable.randomUID("stratifiedSamplingDataBalancer"))

  setDefault(seed, this.getClass.getName.hashCode.toLong)

  def setStrategy(value: String): this.type = set(strategy, value)
  setDefault(strategy, "oversampling")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): DataBalancerModel = {
    validateAndTransformSchema(dataset.schema)
    val counts = dataset.select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
      .toSeq.sortBy(-_._2)
    val factors = counts.map { case (value, count) =>
      (value, counts(0)._2 / count.toDouble)
    }.toArray
    copyValues(new DataBalancerModel(uid, factors).setParent(this))
  }

  override def transformSchema(schema: StructType) = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): DataBalancer = defaultCopy(extra)
}

class DataBalancerModel(
    override val uid: String,
    val factors: Array[(String, Double)])
  extends Model[DataBalancerModel] with DataBalancerBase {

  def this(factor: Array[(String, Double)]) =
    this(Identifiable.randomUID("stratifiedSamplingDataBalancer"), factor)

  setDefault(strategy, "oversampling")
  setDefault(seed, this.getClass.getName.hashCode.toLong)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val datasets = ArrayBuffer[(DataFrame, Double)]()

    var all = dataset.filter(col($(inputCol)).cast(StringType) === factors(0)._1).toDF()
    for (i <- 1 to factors.length - 1) {
      val filteredDataset =
        dataset.filter(col($(inputCol)).cast(StringType) === factors(i)._1).toDF()
      val factor = factors(i)._2
      for (_ <- 1 to factor.toInt) all = all.unionAll(filteredDataset)
      all = all.unionAll(filteredDataset.sample(false, factor % 1))
    }

    all
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): DataBalancerModel = {
    val copied = new DataBalancerModel(uid, factors)
    copyValues(copied, extra).setParent(parent)
  }
}

