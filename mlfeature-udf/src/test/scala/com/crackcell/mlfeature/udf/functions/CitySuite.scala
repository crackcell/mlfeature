package com.crackcell.mlfeature.udf.functions

import org.scalatest.FunSuite

/**
  * Created by Menglong TAN on 3/24/17.
  */
class CitySuite extends FunSuite {

  test("city functions") {
    assert(City.nameToAbbr("北京") === Some("bj"))
    assert(City.abbrToName("bj") === Some("北京"))
  }

}
