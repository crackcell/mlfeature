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

import scala.language.existentials

import org.apache.hadoop.fs.Path

import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.{Estimator, Model, Transformer}
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
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
   * Param for how to handle invalid data (unseen labels or NULL values).
   * Options are 'skip' (filter out rows with invalid data),
   * 'error' (throw an error), or 'keep' (put invalid data in a special additional
   * bucket, at index numLabels.
   * Default: "error"
   * @group param
   */
  @Since("1.6.0")
  val handleInvalid: Param[String] = new Param[String](this, "handleInvalid", "how to handle " +
    "invalid data (unseen labels or NULL values). " +
    "Options are 'skip' (filter out rows with invalid data), error (throw an error), " +
    "or 'keep' (put invalid data in a special additional bucket, at index numLabels).",
    ParamValidators.inArray(MyStringIndexer.supportedHandleInvalids))

  setDefault(handleInvalid, MyStringIndexer.ERROR_INVALID)

  /** @group getParam */
  @Since("1.6.0")
  def getHandleInvalid: String = $(handleInvalid)

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
@Since("1.4.0")
class MyStringIndexer @Since("1.4.0")(
    @Since("1.4.0") override val uid: String) extends Estimator[MyStringIndexerModel]
  with MyStringIndexerBase with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("strIdx"))

  /** @group setParam */
  @Since("1.6.0")
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): MyStringIndexerModel = {
    transformSchema(dataset.schema, logging = true)
    val counts = dataset.na.drop(Array($(inputCol))).select(col($(inputCol)).cast(StringType))
      .rdd
      .map(_.getString(0))
      .countByValue()
    val labels = counts.toSeq.sortBy(-_._2).map(_._1).toArray
    copyValues(new MyStringIndexerModel(uid, labels).setParent(this))
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): MyStringIndexer = defaultCopy(extra)
}

@Since("1.6.0")
object MyStringIndexer extends DefaultParamsReadable[MyStringIndexer] {
  private[feature] val SKIP_INVALID: String = "skip"
  private[feature] val ERROR_INVALID: String = "error"
  private[feature] val KEEP_INVALID: String = "keep"
  private[feature] val supportedHandleInvalids: Array[String] =
    Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)

  @Since("1.6.0")
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
@Since("1.4.0")
class MyStringIndexerModel(
                            @Since("1.4.0") override val uid: String,
    @Since("1.5.0") val labels: Array[String])
  extends Model[MyStringIndexerModel] with MyStringIndexerBase with MLWritable {

  import MyStringIndexerModel._

  @Since("1.5.0")
  def this(labels: Array[String]) = this(Identifiable.randomUID("strIdx"), labels)

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

  /** @group setParam */
  @Since("1.6.0")
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    if (!dataset.schema.fieldNames.contains($(inputCol))) {
      logInfo(s"Input column ${$(inputCol)} does not exist during transformation. " +
        "Skip StringIndexerModel.")
      return dataset.toDF
    }
    transformSchema(dataset.schema, logging = true)

    val filteredLabels = getHandleInvalid match {
      case MyStringIndexer.KEEP_INVALID => labels :+ "__unknown"
      case _ => labels
    }

    val metadata = NominalAttribute.defaultAttr
      .withName($(outputCol)).withValues(filteredLabels).toMetadata()
    // If we are skipping invalid records, filter them out.
    val (filteredDataset, keepInvalid) = getHandleInvalid match {
      case MyStringIndexer.SKIP_INVALID =>
        val filterer = udf { label: String =>
          labelToIndex.contains(label)
        }
        (dataset.na.drop(Array($(inputCol))).where(filterer(dataset($(inputCol)))), false)
      case _ => (dataset, getHandleInvalid == MyStringIndexer.KEEP_INVALID)
    }

    val indexer = udf { row: Row =>
      if (row.isNullAt(0)) {
        if (keepInvalid) {
          labels.length
        } else {
          throw new SparkException("StringIndexer encountered NULL value. To handle or skip " +
            "NULLS, try setting StringIndexer.handleInvalid.")
        }
      } else {
        val label = String.valueOf(row.get(0))
        if (labelToIndex.contains(label)) {
          labelToIndex(label)
        } else if (keepInvalid) {
          labels.length
        } else {
          throw new SparkException(s"Unseen label: $label.  To handle unseen labels, " +
            s"set Param handleInvalid to ${MyStringIndexer.KEEP_INVALID}.")
        }
      }
    }

    filteredDataset.select(col("*"),
      indexer(struct(Array(dataset($(inputCol))): _*)).as($(outputCol), metadata))
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(inputCol))) {
      validateAndTransformSchema(schema)
    } else {
      // If the input column does not exist during transformation, we skip StringIndexerModel.
      schema
    }
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): MyStringIndexerModel = {
    val copied = new MyStringIndexerModel(uid, labels)
    copyValues(copied, extra).setParent(parent)
  }

  @Since("1.6.0")
  override def write: MyStringIndexModelWriter = new MyStringIndexModelWriter(this)
}

@Since("1.6.0")
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

  private class StringIndexerModelReader extends MLReader[MyStringIndexerModel] {

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

  @Since("1.6.0")
  override def read: MLReader[MyStringIndexerModel] = new StringIndexerModelReader

  @Since("1.6.0")
  override def load(path: String): MyStringIndexerModel = super.load(path)
}
