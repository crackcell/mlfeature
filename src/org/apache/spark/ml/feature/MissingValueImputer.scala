package org.apache.spark.ml.feature

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol, HasStrategy}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
  * Created by Menglong TAN on 1/19/17.
  */
class MissingValueImputer(override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with HasStrategy with DefaultParamsWritable {

  val supportedMethods = Array[String]("mean")

  def this() = this(Identifiable.randomUID("missValueMean"))

  setDefault(strategy, "mean")

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setMethod(value: String): this.type = set(strategy, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    $(strategy) match {
      case "mean" => withMeanMethod(dataset)
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    require(supportedMethods.contains($(strategy)), s"method not supported:  ${$(strategy)}")
    SchemaUtils.checkColumnTypes(schema, $(inputCol), Seq(IntegerType, FloatType, DoubleType))
    require(!schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), DoubleType, false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): MissingValueImputer = defaultCopy(extra)

  private def withMeanMethod(dataset: Dataset[_]): DataFrame = {
    val v = dataset.agg(Map($(inputCol) -> "sum")).collect()(0)
    val sum =
      dataset.schema($(inputCol)).dataType match {
        case _: IntegerType => v.getLong(0)
        case _: FloatType => v.getFloat(0)
        case _: DoubleType => v.getDouble(0)
      }

    val count = dataset.filter(s"${$(inputCol)} IS NOT NULL").count()
    val mean = sum / count
    val attr = NumericAttribute.defaultAttr.withName($(outputCol))
    transformSchema(dataset.schema, logging = true)
    dataset.select(
      col("*"),
      when(
        col($(inputCol)).isNull,
        mean
      ).otherwise(col($(inputCol))).as($(outputCol), attr.toMetadata())
    )
  }
}