package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
  * Created by Menglong TAN on 1/19/17.
  */
trait MissingValueMeanImputerParams extends Params with HasInputCol with HasOutputCol {
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnTypes(schema, $(inputCol), Seq(IntegerType, FloatType, DoubleType))
    require(!schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ StructField($(outputCol), DoubleType, false)
    StructType(outputFields)
  }
}

class MissingValueMeanImputer(override val uid: String)
  extends Estimator[MissingValueMeanImputerModel] with MissingValueMeanImputerParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("missValueMean"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): MissingValueMeanImputerModel = {
    transformSchema(dataset.schema, logging = true)
    val mean = calcMean(dataset)
    copyValues(new MissingValueMeanImputerModel(uid, mean).setParent(this))
  }

  private def calcMean(dataset: Dataset[_]): Double = {
    val v = dataset.agg(Map($(inputCol) -> "sum")).collect()(0)
    val sum =
      dataset.schema($(inputCol)).dataType match {
        case _: IntegerType => v.getLong(0)
        case _: FloatType => v.getFloat(0)
        case _: DoubleType => v.getDouble(0)
      }

    val count = dataset.filter(s"${$(inputCol)} IS NOT NULL").count()
    val mean = sum / count
    mean
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): MissingValueMeanImputer = defaultCopy(extra)

}

class MissingValueMeanImputerModel(override val uid: String, val originalMean: Double)
  extends Model[MissingValueMeanImputerModel] with MissingValueMeanImputerParams with MLWritable {

  import MissingValueMeanImputerModel._

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val attr = NumericAttribute.defaultAttr.withName($(outputCol))
    dataset.select(
      col("*"),
      when(
        col($(inputCol)).isNull,
        originalMean
      ).otherwise(col($(inputCol))).as($(outputCol), attr.toMetadata())
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): MissingValueMeanImputerModel = {
    val copied = new MissingValueMeanImputerModel(uid, originalMean)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new MissingValueMeanImputerModelWriter(this)
}

object MissingValueMeanImputerModel extends MLReadable[MissingValueMeanImputerModel] {

  override def read: MLReader[MissingValueMeanImputerModel] = new MissingValueMeanImputerModelReader

  override def load(path: String): MissingValueMeanImputerModel = super.load(path)

  private[MissingValueMeanImputerModel]
  class MissingValueMeanImputerModelWriter(instance: MissingValueMeanImputerModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = new Data(instance.originalMean)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }

    private case class Data(originalMean: Double)
  }

  private class MissingValueMeanImputerModelReader extends MLReader[MissingValueMeanImputerModel] {

    private val className = classOf[MissingValueMeanImputerModel].getName

    override def load(path: String): MissingValueMeanImputerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
        .select("originalMean")
        .head()
      val originalMean = data.getAs[Double](0)
      val model = new MissingValueMeanImputerModel(metadata.uid, originalMean)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
