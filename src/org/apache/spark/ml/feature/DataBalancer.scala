package org.apache.spark.ml.feature

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasLabelCol, HasStrategy, HasSeed}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructType}

import scala.collection.mutable.ArrayBuffer

class DataBalancer(override val uid: String)
  extends Transformer with HasStrategy with HasLabelCol with HasSeed with DefaultParamsWritable {

  setDefault(strategy, "random")
  setDefault(seed, this.getClass.getName.hashCode.toLong)

  def this() = this(Identifiable.randomUID("oversampler"))

  def setStrategy(value: String): this.type = set(strategy, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setSeed(value: Long): this.type = set(seed, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    $(strategy) match {
      case "random" => withRandomStrategy(dataset)
    }
  }

  override def copy(extra: ParamMap): DataBalancer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    require(get(labelCol).isDefined, "Label col must be defined first.")
    SchemaUtils.checkColumnTypes(schema, $(labelCol), Seq(IntegerType, DoubleType))
    StructType(schema.fields)
  }

  private def withRandomStrategy(dataset: Dataset[_]): DataFrame = {
    val pos = dataset.filter(col($(labelCol)) === 1)
    val neg = dataset.filter(col($(labelCol)) === 0)

    val skew = pos.count / neg.count.toDouble
    val data = if (skew < 1) randomOversample(pos, dataset, skew) else randomOversample(neg, dataset, 1 / skew)
    data.select("*")
  }

  private def randomOversample(smaller: Dataset[_], all: Dataset[_], skew: Double): DataFrame = {
    val datasets = ArrayBuffer[DataFrame]()

    def unionAll(frames: Array[DataFrame]): DataFrame = {
      var all = frames(0)
      for (i <- 1 to frames.length - 1) all = all.unionAll(frames(i))
      all
    }

    for (_ <- 1 to (1 / skew - 1).toInt) datasets += smaller.toDF()
    datasets += smaller.sample(false, 1 / skew % 1, $(seed)).toDF()
    datasets += all.toDF()

    unionAll(datasets.toArray)
  }

}
