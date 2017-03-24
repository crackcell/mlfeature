package org.apache.spark.ml.feature

import org.apache.spark.SparkException
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasSeed}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by Menglong TAN on 3/21/17.
  */
class DataBalancer(override val uid: String)
  extends Transformer with HasInputCol with HasSeed with DefaultParamsWritable {

  val strategy: Param[String] = new Param[String](this, "strategy",
    "how to handle imbalanced dataset. Options are 'oversampling' or 'undersampling'",
    ParamValidators.inArray(DataBalancer.supportedStrategies))

  setDefault(seed, this.getClass.getName.hashCode.toLong)

  def this() = this(Identifiable.randomUID("stratifiedSamplingDataBalancer"))

  def setStrategy(value: String): this.type = set(strategy, value)

  setDefault(strategy, "oversampling")

  def setInputCol(value: String): this.type = set(inputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val counts = dataset.select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
      .toSeq.sortBy(-_._2)

    val factors = $(strategy) match {
      case DataBalancer.OVERSAMPLING_STRATEGY =>
        counts.map { case (value, count) =>
          (value, counts(0)._2 / count.toDouble)
        }.toArray
      case DataBalancer.UNDERSAMPLING_STRATEGY =>
        counts.map { case (value, count) =>
          (value, counts.last._2 / count.toDouble)
        }.toArray.reverse
    }

    var all = dataset.filter(col($(inputCol)).cast(StringType) === factors(0)._1).toDF()
    for (i <- 1 to factors.length - 1) {
      val value = factors(i)._1
      val factor = factors(i)._2
      val filteredDataset =
        dataset.filter(col($(inputCol)).cast(StringType) === value).toDF()
      for (_ <- 1 to factor.toInt) all = all.union(filteredDataset)
      if (factor % 1 != 0) all = all.union(filteredDataset.sample(false, factor % 1))
    }
    all
  }

  def getStrategy: String = $(strategy)

  override def transformSchema(schema: StructType) = {
    validateAndTransformSchema(schema)
  }

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(schema.fieldNames.contains($(inputCol)), s"Input column ${$(inputCol)} does not exist")
    schema
  }

  override def copy(extra: ParamMap): DataBalancer = defaultCopy(extra)

}

private object DataBalancer extends DefaultParamsReadable[DataBalancer] {
  private[feature] val OVERSAMPLING_STRATEGY: String = "oversampling"
  private[feature] val UNDERSAMPLING_STRATEGY: String = "undersampling"
  private[feature] val supportedStrategies: Array[String] =
    Array(OVERSAMPLING_STRATEGY, UNDERSAMPLING_STRATEGY)

  override def load(path: String): DataBalancer = super.load(path)
}


