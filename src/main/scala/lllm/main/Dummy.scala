package lllm.main

import breeze.linalg.DenseVector
import breeze.numerics.{exp, log}
import breeze.optimize.{GradientTester, DiffFunction}
import breeze.optimize.FirstOrderMinimizer.OptParams

/**
 * @author jda
 */
object Dummy {


  def main(args: Array[String]): Unit = {
    val objective = new DiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]): (Double, DenseVector[Double]) = {
        val t1a = x(0)
        val t1b = x(1)
        val t2a = x(2)
        val t2b = x(3)

        val ll = 0.5 * (t1a + t1b) - log(exp(t1a) + exp(t1b)) +
          10 * (t1b + t2b - log(exp(t1a + t2a) + exp(t1b + t2b))) +
          t2a - log(exp(t2a) + exp(t2b))

        val z1 = exp(t1a) + exp(t1b)
        val z12 = exp(t1a + t2a) + exp(t1b + t2b)
        val z2 = exp(t2a) + exp(t2b)

        val g1a = 0.5 - exp(t1a) / z1 - exp(t1a + t2a) / z12
        val g1b = 0.5 - exp(t1b) / z1 + 1d - exp(t1b + t2b) / z12

        val g2a = -exp(t1a + t2a) / z12 + 1d - exp(t2a) / z2
        val g2b = 1d - exp(t1b + t2b) / z12 - exp(t2b) / z2

        val grad = DenseVector(g1a, g1b, g2a, g2b)
        (-ll, -grad)
      }
    }

    GradientTester.test(objective, DenseVector.zeros[Double](4), randFraction=0.5, skipZeros=false, toString = (x:Int) => x.toString)
//    println("tested")
    val opt = OptParams(maxIterations = 100, regularization=0)
    val optParams = opt.minimize(objective, DenseVector.zeros[Double](4))
    println(optParams)
  }

}
