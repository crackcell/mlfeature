package org.apache.spark.ml.feature

import org.apache.spark.{MyTestSuite, SparkException}
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._

/**
  * Created by crackcell on 3/13/17.
  */
class MyStringIndexerSuite extends MyTestSuite {

  import sqlContext.implicits._

  test("StringIndexer") {
    val data = Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    val df = data.toDF("id", "label")
    val indexer = new MyStringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
      .fit(df)

    val transformed = indexer.transform(df)
    val attr = Attribute.fromStructField(transformed.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attr.values.get === Array("a", "c", "b"))
    val output = transformed.select("id", "labelIndex").rdd.map { r =>
      (r.getInt(0), r.getDouble(1))
    }.collect().toSet
    // a -> 0, b -> 2, c -> 1
    val expected = Set((0, 0.0), (1, 2.0), (2, 1.0), (3, 0.0), (4, 0.0), (5, 1.0))
    assert(output === expected)
  }

  test("StringIndexerUnseen") {
    val data = Seq((0, "a"), (1, "b"), (4, "b"))
    val data2 = Seq((0, "a"), (1, "b"), (2, "c"), (3, "d"))
    val df = data.toDF("id", "label")
    val df2 = data2.toDF("id", "label")
    val indexer = new MyStringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
      .fit(df)
    // Verify we throw by default with unseen values
    intercept[SparkException] {
      indexer.transform(df2).collect()
    }

    indexer.setHandleInvalid("skip")
    // Verify that we skip the c record
    val transformedSkip = indexer.transform(df2)
    val attrSkip = Attribute.fromStructField(transformedSkip.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attrSkip.values.get === Array("b", "a"))
    val outputSkip = transformedSkip.select("id", "labelIndex").rdd.map { r =>
      (r.getInt(0), r.getDouble(1))
    }.collect().toSet
    // a -> 1, b -> 0
    val expectedSkip = Set((0, 1.0), (1, 0.0))
    assert(outputSkip === expectedSkip)

    indexer.setHandleInvalid("keep")
    // Verify that we keep the unseen records
    val transformedKeep = indexer.transform(df2)
    val attrKeep = Attribute.fromStructField(transformedKeep.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attrKeep.values.get === Array("b", "a", "__unknown"))
    val outputKeep = transformedKeep.select("id", "labelIndex").rdd.map { r =>
      (r.getInt(0), r.getDouble(1))
    }.collect().toSet
    // a -> 1, b -> 0, c -> 2, d -> 3
    val expectedKeep = Set((0, 1.0), (1, 0.0), (2, 2.0), (3, 2.0))
    assert(outputKeep === expectedKeep)
  }

  test("StringIndexer with a numeric input column") {
    val data = Seq((0, 100), (1, 200), (2, 300), (3, 100), (4, 100), (5, 300))
    val df = data.toDF("id", "label")
    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
      .fit(df)
    val transformed = indexer.transform(df)
    val attr = Attribute.fromStructField(transformed.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attr.values.get === Array("100", "300", "200"))
    val output = transformed.select("id", "labelIndex").rdd.map { r =>
      (r.getInt(0), r.getDouble(1))
    }.collect().toSet
    // 100 -> 0, 200 -> 2, 300 -> 1
    val expected = Set((0, 0.0), (1, 2.0), (2, 1.0), (3, 0.0), (4, 0.0), (5, 1.0))
    assert(output === expected)
  }

  test("StringIndexer with a string input column with NULLs") {
    val data: Seq[java.lang.String] = Seq("a", "b", "b", null)
    val data2: Seq[java.lang.String] = Seq("a", "b", null)
    val expectedSkip = Array(1.0, 0.0)
    val expectedKeep = Array(1.0, 0.0, 2.0)
    val df = data.toDF("label")
    val df2 = data2.toDF("label")

    val indexer = new MyStringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")

    withClue("StringIndexer should throw error when setHandleValid=error when given NULL values") {
      intercept[SparkException] {
        indexer.setHandleInvalid("error")
        indexer.fit(df).transform(df2).collect()
      }
    }

    indexer.setHandleInvalid("skip")
    val transformedSkip = indexer.fit(df).transform(df2)
    val attrSkip = Attribute
      .fromStructField(transformedSkip.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attrSkip.values.get === Array("b", "a"))
    assert(transformedSkip.select("labelIndex").rdd.map { r =>
      r.getDouble(0)
    }.collect() === expectedSkip)

    indexer.setHandleInvalid("keep")
    val transformedKeep = indexer.fit(df).transform(df2)
    val attrKeep = Attribute
      .fromStructField(transformedKeep.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attrKeep.values.get === Array("b", "a", "__unknown"))
    assert(transformedKeep.select("labelIndex").rdd.map { r =>
      r.getDouble(0)
    }.collect() === expectedKeep)
  }

  test("StringIndexer with a numeric input column with NULLs") {
    val data: Seq[Integer] = Seq(1, 2, 2, null)
    val data2: Seq[Integer] = Seq(1, 2, null)
    val expectedSkip = Array(1.0, 0.0)
    val expectedKeep = Array(1.0, 0.0, 2.0)
    val df = data.toDF("label")
    val df2 = data2.toDF("label")

    val indexer = new MyStringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")

    withClue("StringIndexer should throw error when setHandleValid=error when given NULL values") {
      intercept[SparkException] {
        indexer.setHandleInvalid("error")
        indexer.fit(df).transform(df2).collect()
      }
    }

    indexer.setHandleInvalid("skip")
    val transformedSkip = indexer.fit(df).transform(df2)
    val attrSkip = Attribute
      .fromStructField(transformedSkip.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attrSkip.values.get === Array("2", "1"))
    assert(transformedSkip.select("labelIndex").rdd.map { r =>
      r.getDouble(0)
    }.collect() === expectedSkip)

    indexer.setHandleInvalid("keep")
    val transformedKeep = indexer.fit(df).transform(df2)
    val attrKeep = Attribute
      .fromStructField(transformedKeep.schema("labelIndex"))
      .asInstanceOf[NominalAttribute]
    assert(attrKeep.values.get === Array("2", "1", "__unknown"))
    assert(transformedKeep.select("labelIndex").rdd.map { r =>
      r.getDouble(0)
    }.collect() === expectedKeep)
  }

  test("StringIndexerModel can't overwrite output column") {
    val df = Seq((1, 2), (3, 4)).toDF("input", "output")
    intercept[IllegalArgumentException] {
      new StringIndexer()
        .setInputCol("input")
        .setOutputCol("output")
        .fit(df)
    }

    val indexer = new MyStringIndexer()
      .setInputCol("input")
      .setOutputCol("indexedInput")
      .fit(df)

    intercept[IllegalArgumentException] {
      indexer.setOutputCol("output").transform(df)
    }
  }

  test("IndexToString.transform") {
    val labels = Array("a", "b", "c")
    val df0 = Seq((0, "a"), (1, "b"), (2, "c"), (0, "a")).toDF("index", "expected")

    val idxToStr0 = new IndexToString()
      .setInputCol("index")
      .setOutputCol("actual")
      .setLabels(labels)
    idxToStr0.transform(df0).select("actual", "expected").collect().foreach {
      case Row(actual, expected) =>
        assert(actual === expected)
    }

    val attr = NominalAttribute.defaultAttr.withValues(labels)
    val df1 = df0.select(col("index").as("indexWithAttr", attr.toMetadata()), col("expected"))

    val idxToStr1 = new IndexToString()
      .setInputCol("indexWithAttr")
      .setOutputCol("actual")
    idxToStr1.transform(df1).select("actual", "expected").collect().foreach {
      case Row(actual, expected) =>
        assert(actual === expected)
    }
  }

  test("StringIndexer, IndexToString are inverses") {
    val data = Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    val df = data.toDF("id", "label")
    val indexer = new MyStringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
      .fit(df)
    val transformed = indexer.transform(df)
    val idx2str = new IndexToString()
      .setInputCol("labelIndex")
      .setOutputCol("sameLabel")
      .setLabels(indexer.labels)
    idx2str.transform(transformed).select("label", "sameLabel").collect().foreach {
      case Row(a: String, b: String) =>
        assert(a === b)
    }
  }

  test("IndexToString.transformSchema (SPARK-10573)") {
    val idxToStr = new IndexToString().setInputCol("input").setOutputCol("output")
    val inSchema = StructType(Seq(StructField("input", DoubleType)))
    val outSchema = idxToStr.transformSchema(inSchema)
    assert(outSchema("output").dataType === StringType)
  }

  test("SPARK 18698: construct IndexToString with custom uid") {
    val uid = "customUID"
    val t = new IndexToString(uid)
    assert(t.uid == uid)
  }

  test("StringIndexer metadata") {
    val data = Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    val df = data.toDF("id", "label")
    val indexer = new MyStringIndexer()
      .setInputCol("label")
      .setOutputCol("labelIndex")
      .fit(df)
    val transformed = indexer.transform(df)
    val attrs =
      NominalAttribute.decodeStructField(transformed.schema("labelIndex"), preserveName = true)
    assert(attrs.name.nonEmpty && attrs.name.get === "labelIndex")
  }
}
