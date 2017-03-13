package org.apache.spark

import org.apache.spark.sql.SQLContext
import org.scalatest.FunSuite

/**
  * Created by crackcell on 3/13/17.
  */
class MyTestSuite extends FunSuite {
  val conf = new SparkConf().setAppName("test").setMaster("local[*]")
  val sc = SparkContext.getOrCreate(conf)
  sc.setLogLevel("warn")
  val sqlContext = new SQLContext(sc)
}
