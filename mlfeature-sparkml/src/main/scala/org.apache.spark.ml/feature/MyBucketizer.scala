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

import java.{util => ju}

import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.ml.Model
import org.apache.spark.ml.attribute.NominalAttribute
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}

/**
  * `Bucketizer` maps a column of continuous features to a column of feature buckets.
  */
@Since("1.4.0")
final class MyBucketizer @Since("1.4.0")(@Since("1.4.0") override val uid: String)
  extends Model[MyBucketizer] with HasInputCol with HasOutputCol with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("bucketizer"))

  /**
    * Parameter for mapping continuous features into buckets. With n+1 splits, there are n buckets.
    * A bucket defined by splits x,y holds values in the range [x,y) except the last bucket, which
    * also includes y. Splits should be of length greater than or equal to 3 and strictly increasing.
    * Values at -inf, inf must be explicitly provided to cover all Double values;
    * otherwise, values outside the splits specified will be treated as errors.
    *
    * See also [[handleInvalid]], which can optionally create an additional bucket for NaN/NULL
    * values.
    *
    * @group param
    */
  @Since("1.4.0")
  val splits: DoubleArrayParam = new DoubleArrayParam(this, "splits",
    "Split points for mapping continuous features into buckets. With n+1 splits, there are n " +
      "buckets. A bucket defined by splits x,y holds values in the range [x,y) except the last " +
      "bucket, which also includes y. The splits should be of length >= 3 and strictly " +
      "increasing. Values at -inf, inf must be explicitly provided to cover all Double values; " +
      "otherwise, values outside the splits specified will be treated as errors.",
    MyBucketizer.checkSplits)

  /** @group getParam */
  @Since("1.4.0")
  def getSplits: Array[Double] = $(splits)

  /** @group setParam */
  @Since("1.4.0")
  def setSplits(value: Array[Double]): this.type = set(splits, value)

  /** @group setParam */
  @Since("1.4.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.4.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
    * Param for how to handle invalid entries. Options are 'skip' (filter out rows with
    * invalid values), 'error' (throw an error), or 'keep' (keep invalid values in a special
    * additional bucket).
    * Default: "error"
    * @group param
    */
  // TODO: SPARK-18619 Make Bucketizer inherit from HasHandleInvalid.
  @Since("2.1.0")
  val handleInvalid: Param[String] = new Param[String](this, "handleInvalid", "how to handle " +
    "invalid entries. Options are skip (filter out rows with invalid values), " +
    "error (throw an error), or keep (keep invalid values in a special additional bucket).",
    ParamValidators.inArray(MyBucketizer.supportedHandleInvalids))

  /** @group getParam */
  @Since("2.1.0")
  def getHandleInvalid: String = $(handleInvalid)

  /** @group setParam */
  @Since("2.1.0")
  def setHandleInvalid(value: String): this.type = set(handleInvalid, value)
  setDefault(handleInvalid, MyBucketizer.ERROR_INVALID)

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    val (filteredDataset, keepInvalid) = {
      if (getHandleInvalid == MyBucketizer.SKIP_INVALID) {
        // "skip" NaN/NULL option is set, will filter out NaN/NULL values in the dataset
        var filteredDataset = dataset.na.drop(Array($(inputCol)))
        if ($(splits)(0) != Double.NegativeInfinity) {
          filteredDataset = filteredDataset.filter(s"${$(inputCol)} > ${$(splits)(0)}")
        }
        if ($(splits).last != Double.PositiveInfinity) {
          filteredDataset = filteredDataset.filter(s"${$(inputCol)} <= ${$(splits).last}")
        }
        (filteredDataset.toDF(), false)
      } else {
        (dataset.toDF(), getHandleInvalid == MyBucketizer.KEEP_INVALID)
      }
    }

    val bucketizer: UserDefinedFunction = udf { (row: Row) =>
      val feature = if (row.isNullAt(0)) None else Some(row.getDouble(0))
      MyBucketizer.binarySearchForBuckets($(splits), feature, keepInvalid)
    }

    val newCol = bucketizer(struct(Array(filteredDataset($(inputCol))): _*))
    val newField = prepOutputField(filteredDataset.schema)
    filteredDataset.select(col("*"), newCol.as($(outputCol), newField.metadata))
  }

  private def prepOutputField(schema: StructType): StructField = {
    val buckets = $(splits).sliding(2).map(bucket => bucket.mkString(", ")).toArray
    val attr = new NominalAttribute(name = Some($(outputCol)), isOrdinal = Some(true),
      values = Some(buckets))
    attr.toStructField()
  }

  @Since("1.4.0")
  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), DoubleType)
    SchemaUtils.appendColumn(schema, prepOutputField(schema))
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): MyBucketizer = {
    defaultCopy[MyBucketizer](extra).setParent(parent)
  }
}

@Since("1.6.0")
object MyBucketizer extends DefaultParamsReadable[MyBucketizer] {

  private[feature] val SKIP_INVALID: String = "skip"
  private[feature] val ERROR_INVALID: String = "error"
  private[feature] val KEEP_INVALID: String = "keep"
  private[feature] val supportedHandleInvalids: Array[String] =
    Array(SKIP_INVALID, ERROR_INVALID, KEEP_INVALID)

  /**
    * We require splits to be of length >= 3 and to be in strictly increasing order.
    * No NaN split should be accepted.
    */
  private[feature] def checkSplits(splits: Array[Double]): Boolean = {
    if (splits.length < 3) {
      false
    } else {
      var i = 0
      val n = splits.length - 1
      while (i < n) {
        if (splits(i) >= splits(i + 1) || splits(i).isNaN) return false
        i += 1
      }
      !splits(n).isNaN
    }
  }

  /**
    * Binary searching in several buckets to place each data point.
    * @param splits array of split points
    * @param feature data point
    * @param keepInvalid NaN/NULL flag.
    *                    Set "true" to make an extra bucket for NaN/NULL values;
    *                    Set "false" to report an error for NaN/NULL values
    * @return bucket for each data point
    * @throws SparkException if a feature is < splits.head or > splits.last
    */

  private[feature] def binarySearchForBuckets(
      splits: Array[Double],
      feature: Option[Double],
      keepInvalid: Boolean): Double = {
    if (feature.getOrElse(Double.NaN).isNaN) {
      if (keepInvalid) {
        splits.length - 1
      } else {
        throw new SparkException("Bucketizer encountered NaN/NULL values. " +
          "To handle or skip NaNs/NULLs, try setting Bucketizer.handleInvalid.")
      }
    } else if (feature.get == splits.last) {
      splits.length - 2
    } else {
      val idx = ju.Arrays.binarySearch(splits, feature.get)
      if (idx >= 0) {
        idx
      } else {
        val insertPos = -idx - 1
        if (insertPos == 0 || insertPos == splits.length) {
          if (keepInvalid) {
            splits.length - 1
          } else {
            throw new SparkException(s"Feature value ${feature.get} out of Bucketizer bounds" +
              s" [${splits.head}, ${splits.last}].  Check your features, or loosen " +
              s"the lower/upper bound constraints.")
          }
        } else {
          insertPos - 1
        }
      }
    }
  }

  @Since("1.6.0")
  override def load(path: String): MyBucketizer = super.load(path)
}