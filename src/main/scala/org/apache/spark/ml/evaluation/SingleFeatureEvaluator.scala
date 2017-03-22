package org.apache.spark.ml.evaluation

import org.apache.spark.SparkException
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol}
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset

/**
  * Created by Menglong TAN on 3/22/17.
  */
class SingleFeatureEvaluator(override val uid: String)
  extends Evaluator with HasInputCol with HasLabelCol with DefaultParamsWritable {

  val metricName = new Param[String](this, "metricName",
    "metric name in evaluation (areaUnderROC)",
    ParamValidators.inArray(SingleFeatureEvaluator.supportedMetrics))

  def this() = this(Identifiable.randomUID("singleFeaEval"))

  def getMetricName: String = $(metricName)

  def setMetricName(value: String): this.type = set(metricName, value)

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setLabelCol(value: String): this.type = set(labelCol, value)

  setDefault(metricName, SingleFeatureEvaluator.AUROC)

  override def evaluate(dataset: Dataset[_]): Double = {
    0
  }

  override def isLargerBetter: Boolean = $(metricName) match {
    case SingleFeatureEvaluator.AUROC => true
    case _ => throw new SparkException(s"metric ${$(metricName)} is not supported")
  }

  override def copy(extra: ParamMap): SingleFeatureEvaluator = defaultCopy(extra)
}

object SingleFeatureEvaluator {
  private[feature] val AUROC = "areaUnderROC"
  private[feature] val supportedMetrics = Array(AUROC)
}