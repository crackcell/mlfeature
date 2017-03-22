package org.apache.spark.ml.feature

import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol, HasSeed, HasStrategy}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, MLWritable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by Menglong TAN on 3/21/17.
  */
class DataBalancer(override val uid: String)
  extends Transformer with HasInputCol with HasSeed with DefaultParamsWritable{

  def this() = this(Identifiable.randomUID("stratifiedSamplingDataBalancer"))

  setDefault(seed, this.getClass.getName.hashCode.toLong)

  val strategy: Param[String] = new Param[String](this, "strategy",
    "how to handle imbalanced dataset. Options are oversampling",
    ParamValidators.inArray(DataBalancer.supportedStrategies))

  def setStrategy(value: String): this.type = set(strategy, value)
  setDefault(strategy, "oversampling")

  def getStrategy: String = $(strategy)

  def setInputCol(value: String): this.type = set(inputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val counts = dataset.select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
      .toSeq.sortBy(-_._2)
    val factors = counts.map { case (value, count) =>
      (value, counts(0)._2 / count.toDouble)
    }.toArray

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

  override def transformSchema(schema: StructType) = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): DataBalancer = defaultCopy(extra)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains($(inputCol)), s"Input column ${$(inputCol)} does not exist")
    schema
  }

}

private object DataBalancer extends DefaultParamsReadable[DataBalancer] {
  private[feature] val OVERSAMPLING_STRATEGY: String = "oversampling"
  private[feature] val supportedStrategies: Array[String] = Array(OVERSAMPLING_STRATEGY)

  override def load(path: String): DataBalancer = super.load(path)
}


