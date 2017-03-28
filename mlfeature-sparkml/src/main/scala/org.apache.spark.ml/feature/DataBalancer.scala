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
    "how to handle imbalanced dataset. Options are 'oversampling', 'undersampling' " +
    "or 'midllesampling'",
    ParamValidators.inArray(DataBalancer.supportedStrategies))


  def this() = this(Identifiable.randomUID("stratifiedSamplingDataBalancer"))

  def setStrategy(value: String): this.type = set(strategy, value)

  def setInputCol(value: String): this.type = set(inputCol, value)

  setDefault(strategy, "undersampling")
  setDefault(seed, this.getClass.getName.hashCode.toLong)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val counts = dataset.select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
      .toSeq.sortBy(-_._2)

    val factors: Array[(String, Double)] = $(strategy) match {
      case DataBalancer.OVER_SAMPLING_STRATEGY =>
        counts.map { case (value, count) =>
          (value, counts(0)._2 / count.toDouble)
        }.toArray
      case DataBalancer.UNDER_SAMPLING_STRATEGY =>
        counts.map { case (value, count) =>
          (value, counts.last._2 / count.toDouble)
        }.toArray.reverse
      case DataBalancer.MIDDLE_SAMPLING_STRATEGY =>
        val originalFactors = counts.map { case (value, count) =>
          (value, counts(0)._2 / count.toDouble)
        }.toArray
        if (originalFactors.length < 3) {
          originalFactors
        } else {
          val pivot: Int = originalFactors.length / 2
          originalFactors.slice(0, pivot) ++
            Array(originalFactors(pivot)) ++
              originalFactors.slice(pivot + 1, originalFactors.length)
        }
    }

    var all = dataset.filter(col($(inputCol)).cast(StringType) === factors(0)._1).toDF()
    for (i <- 1 to factors.length - 1) {
      val value = factors(i)._1
      val factor = factors(i)._2
      val filteredDataset =
        dataset.filter(col($(inputCol)).cast(StringType) === value).toDF()
      for (_ <- 1 to factor.toInt) all = all.union(filteredDataset)
      if (factor % 1 != 0) all = all.union(filteredDataset.sample(true, factor % 1, $(seed)))
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
  private[feature] val OVER_SAMPLING_STRATEGY = "oversampling"
  private[feature] val UNDER_SAMPLING_STRATEGY = "undersampling"
  private[feature] val MIDDLE_SAMPLING_STRATEGY = "middlesampling"
  private[feature] val supportedStrategies: Array[String] =
    Array(OVER_SAMPLING_STRATEGY, UNDER_SAMPLING_STRATEGY, MIDDLE_SAMPLING_STRATEGY)

  override def load(path: String): DataBalancer = super.load(path)
}


