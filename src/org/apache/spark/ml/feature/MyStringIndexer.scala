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

import org.apache.hadoop.fs.Path

import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.util.collection.OpenHashMap

/**
  * Base trait for [[MyStringIndexer]] and [[MyStringIndexerModel]].
  */
private[feature] trait MyStringIndexerBase extends Params with HasInputCol with HasOutputCol {

  /**
    * Param for how to handle invalid entries. Options are 'skip' (filter out rows with
    * invalid values), 'error' (throw an error), or 'keep' (keep invalid values in a special
    * additional bucket).
    * Default: "error"
    * @group param
    */
  val handleInvalid: Param[String] = new Param[String](this, "handleInvalid", "how to handle " +
    "invalid entries. Options are skip (filter out rows with invalid values), " +
    "error (throw an error), or keep (give invalid values a special additional index).",
    ParamValidators.inArray(MyStringIndexer.supportedHandleInvalids))

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == StringType || inputDataType.isInstanceOf[NumericType],
      s"The input column $inputColName must be either string type or numeric type, " +
        s"but got $inputDataType.")
    val inputFields = schema.fields
    val outputColName = $(outputCol)
    require(inputFields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    val attr = NominalAttribute.defaultAttr.withName($(outputCol))
    val outputFields = inputFields :+ attr.toStructField()
    StructType(outputFields)
  }
}

/**
  * A label indexer that maps a string column of labels to an ML column of label indices.
  * If the input column is numeric, we cast it to string and index the string values.
  * The indices are in [0, numLabels), ordered by label frequencies.
  * So the most frequent label gets index 0.
  *
  * @see `IndexToString` for the inverse transformation
  */
class MyStringIndexer @Since("1.4.0")(
                                       @Since("1.4.0") override val uid: String) extends Estimator[MyStringIndexerModel]
  with MyStringIndexerBase with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("strIdx"))

  /** @group getParam */
  def getHandleInvalid: String = $(handleInvalid)

  /** @group setParam */
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)
  setDefault(handleInvalid, MyStringIndexer.ERROR_INVALID)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def fit(dataset: Dataset[_]): MyStringIndexerModel = {
    transformSchema(dataset.schema, logging = true)
    val counts = dataset.select(col($(inputCol)).cast(StringType)).where(col($(inputCol)).isNotNull)
      .rdd
      .map(_.getString(0))
      .countByValue()
    val labels = if (getHandleInvalid == MyStringIndexer.KEEP_INVALID)
      counts.toSeq.sortBy(-_._2).map(_._1).toArray ++ Array("null")
    else
      counts.toSeq.sortBy(-_._2).map(_._1).toArray
    copyValues(new MyStringIndexerModel(uid, labels).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): MyStringIndexer = defaultCopy(extra)
}

object MyStringIndexer extends DefaultParamsReadable[MyStringIndexer] {

  private[feature] val SKIP_INVALID: String = "skip"
  private[feature] val ERROR_INVALID: String = "error"
  private[feature] val KEEP_INVALID: String = "keep"
  private[feature] val supportedHandleInvalids: Array[String] =
    Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)


  override def load(path: String): MyStringIndexer = super.load(path)
}

/**
  * Model fitted by [[MyStringIndexer]].
  *
  * @param labels  Ordered list of labels, corresponding to indices to be assigned.
  * @note During transformation, if the input column does not exist,
  * `StringIndexerModel.transform` would return the input dataset unmodified.
  * This is a temporary fix for the case when target labels do not exist during prediction.
  */
class MyStringIndexerModel(override val uid: String,
                           val labels: Array[String])
  extends Model[MyStringIndexerModel] with MyStringIndexerBase with MLWritable {

  import MyStringIndexerModel._

  def this(labels: Array[String]) = this(Identifiable.randomUID("myStrIdx"), labels)

  private val labelToIndex: OpenHashMap[String, Double] = {
    val n = labels.length
    val map = new OpenHashMap[String, Double](n)
    var i = 0
    while (i < n) {
      map.update(labels(i), i)
      i += 1
    }
    map
  }

  /** @group getParam */
  def getHandleInvalid: String = $(handleInvalid)

  /** @group setParam */
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)
  setDefault(handleInvalid, MyStringIndexer.ERROR_INVALID)

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    if (!dataset.schema.fieldNames.contains($(inputCol))) {
      logInfo(s"Input column ${$(inputCol)} does not exist during transformation. " +
        "Skip StringIndexerModel.")
      return dataset.toDF
    }
    transformSchema(dataset.schema, logging = true)

    val indexer = udf { row: Row =>
      if (row.isNullAt(0)) {
        if (getHandleInvalid == MyStringIndexer.KEEP_INVALID) {
          labelToIndex.size - 1
        } else {
          throw new SparkException("StringIndexer encountered NULL value. To handle or skip " +
            "NULLs, try setting StringIndexer.handleInvalid")
        }
      } else {
        val label = row.getString(0)
        if (labelToIndex.contains(label)) {
          labelToIndex(label)
        } else {
          throw new SparkException(s"Unseen label: $label.")
        }
      }
    }

    val metadata = NominalAttribute.defaultAttr
      .withName($(outputCol)).withValues(labels).toMetadata()
    // If we are skipping invalid records, filter them out.
    val filteredDataset = getHandleInvalid match {
      case "skip" =>
        val filterer = udf { label: String =>
          labelToIndex.contains(label)
        }
        dataset.where(filterer(dataset($(inputCol))))
      case _ => dataset
    }
    filteredDataset.select(col("*"),
      indexer(struct(Array(dataset($(inputCol))): _*)).as($(outputCol), metadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(inputCol))) {
      validateAndTransformSchema(schema)
    } else {
      // If the input column does not exist during transformation, we skip StringIndexerModel.
      schema
    }
  }

  override def copy(extra: ParamMap): MyStringIndexerModel = {
    val copied = new MyStringIndexerModel(uid, labels)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MyStringIndexModelWriter = new MyStringIndexModelWriter(this)
}

object MyStringIndexerModel extends MLReadable[MyStringIndexerModel] {

  private[MyStringIndexerModel]
  class MyStringIndexModelWriter(instance: MyStringIndexerModel) extends MLWriter {

    private case class Data(labels: Array[String])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.labels)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class MyStringIndexerModelReader extends MLReader[MyStringIndexerModel] {

    private val className = classOf[MyStringIndexerModel].getName

    override def load(path: String): MyStringIndexerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
        .select("labels")
        .head()
      val labels = data.getAs[Seq[String]](0).toArray
      val model = new MyStringIndexerModel(metadata.uid, labels)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[MyStringIndexerModel] = new MyStringIndexerModelReader

  override def load(path: String): MyStringIndexerModel = super.load(path)
}

/**
  * A `Transformer` that maps a column of indices back to a new column of corresponding
  * string values.
  * The index-string mapping is either from the ML attributes of the input column,
  * or from user-supplied labels (which take precedence over ML attributes).
  *
  * @see `MyStringIndexer` for converting strings into indices
  */
class IndexToString private[ml] (override val uid: String)
  extends Transformer with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() =
    this(Identifiable.randomUID("myIdxToStr"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setLabels(value: Array[String]): this.type = set(labels, value)

  /**
    * Optional param for array of labels specifying index-string mapping.
    *
    * Default: Not specified, in which case [[inputCol]] metadata is used for labels.
    * @group param
    */
  final val labels: StringArrayParam = new StringArrayParam(this, "labels",
    "Optional array of labels specifying index-string mapping." +
      " If not provided or if empty, then metadata from inputCol is used instead.")

  /** @group getParam */
  final def getLabels: Array[String] = $(labels)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val inputDataType = schema(inputColName).dataType
    require(inputDataType.isInstanceOf[NumericType],
      s"The input column $inputColName must be a numeric type, " +
        s"but got $inputDataType.")
    val inputFields = schema.fields
    val outputColName = $(outputCol)
    require(inputFields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    val outputFields = inputFields :+ StructField($(outputCol), StringType)
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val inputColSchema = dataset.schema($(inputCol))
    // If the labels array is empty use column metadata
    val values = if (!isDefined(labels) || $(labels).isEmpty) {
      Attribute.fromStructField(inputColSchema)
        .asInstanceOf[NominalAttribute].values.get
    } else {
      $(labels)
    }
    val indexer = udf { index: Double =>
      val idx = index.toInt
      if (0 <= idx && idx < values.length) {
        values(idx)
      } else {
        throw new SparkException(s"Unseen index: $index ??")
      }
    }
    val outputColName = $(outputCol)
    dataset.select(col("*"),
      indexer(dataset($(inputCol)).cast(DoubleType)).as(outputColName))
  }

  override def copy(extra: ParamMap): IndexToString = {
    defaultCopy(extra)
  }
}

object IndexToString extends DefaultParamsReadable[IndexToString] {

  override def load(path: String): IndexToString = super.load(path)
}