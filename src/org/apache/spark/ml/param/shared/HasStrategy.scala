package org.apache.spark.ml.param.shared

import org.apache.spark.ml.param.{Param, Params}

/**
  * Created by Menglong TAN on 1/19/17.
  */
private[ml] trait HasStrategy extends Params {
  final val strategy: Param[String] = new Param[String](this, "strategy", "strategy to use")

  /** @group getParam */
  final def getStrategy: String = $(strategy)
}