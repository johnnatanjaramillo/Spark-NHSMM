package org.com.jonas.nhsmm

import scala.util.control.Breaks._
import breeze.linalg.{DenseMatrix, DenseVector, Transpose, normalize, sum}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.com.jonas.nhsmm

object BaumWelchAlgorithm {

  /** * function with reduce function ****/
  def run1(observations: DataFrame, M: Int, k: Int, D: Int,
           initialPi: DenseVector[Double], initialA: DenseVector[DenseMatrix[Double]], initialB: DenseMatrix[Double],
           numPartitions: Int = 1, epsilon: Double = 0.0001, maxIterations: Int = 10000,
           kfold: Int, path_Class_baumwelch: String):
  (DenseVector[Double], DenseVector[DenseMatrix[Double]], DenseMatrix[Double]) = {

    var prior = initialPi
    var transmat = initialA
    var obsmat = initialB
    var antloglik: Double = Double.NegativeInfinity
    val log = org.apache.log4j.LogManager.getRootLogger

    var arrTransmat: Array[Double] = Array()
    (0 until D).foreach(d => arrTransmat = arrTransmat ++ transmat(d).toArray)

    var inInter = 0
    if (new java.io.File(path_Class_baumwelch + kfold).exists) {
      inInter = scala.io.Source.fromFile(path_Class_baumwelch + kfold).getLines.size - 1
      val stringModel: List[String] = scala.io.Source.fromFile(path_Class_baumwelch + kfold).getLines().toList
      val arraymodel = stringModel.last.split(";")

      arrTransmat = arraymodel(5).split(",").map(_.toDouble)

      transmat = DenseVector.fill(D) {
        DenseMatrix.zeros[Double](M, M)
      }

      val multM = M * M
      (0 until D).foreach(d => transmat(d) = new DenseMatrix(M, M, arrTransmat.slice(d * multM, (d + 1) * multM)))

      prior = new DenseVector(arraymodel(4).split(",").map(_.toDouble))
      obsmat = new DenseMatrix(M, k, arraymodel(6).split(",").map(_.toDouble))
      antloglik = arraymodel(7).toDouble

    } else nhsmm.Utils.writeresult(path_Class_baumwelch + kfold, "kfold;iteration;M;k;Pi;A;B;loglik\n")

    observations.persist()
    var obstrained = observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("D", lit(D))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(arrTransmat))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))

    breakable {
      (inInter until maxIterations).foreach(it => {
        log.info("-----------------------------------------------------------------------------------------")
        log.info("Start Iteration: " + it)
        val newvalues = obstrained.repartition(numPartitions)
          .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
          .withColumn("fwdback", udf_fwdback(col("M"), col("k"), col("D"), col("T"), col("Pi"), col("A"), col("obslik"), col("obs")))
          .withColumn("loglik", udf_loglik(col("fwdback")))
          .withColumn("prior", udf_newPi(col("fwdback")))
          .withColumn("transmat", udf_newA(col("fwdback")))
          .withColumn("obsmat", udf_newB(col("fwdback")))
          .drop("workitem", "str_obs", "M", "k", "D", "Pi", "A", "B", "P", "obs", "T", "obslik", "fwdback")
          .reduce((row1, row2) =>
            Row(row1.getAs[Double](0) + row2.getAs[Double](0),
              (row1.getAs[Seq[Double]](1), row2.getAs[Seq[Double]](1)).zipped.map(_ + _),
              (row1.getAs[Seq[Double]](2), row2.getAs[Seq[Double]](2)).zipped.map(_ + _),
              (row1.getAs[Seq[Double]](3), row2.getAs[Seq[Double]](3)).zipped.map(_ + _)))

        val loglik = newvalues.getAs[Double](0)
        log.info("LogLikehood Value: " + loglik)

        val multM = M * M
        var arrTransmat2: Array[Double] = newvalues.getAs[Seq[Double]](2).toArray
        (0 until D).foreach(d => transmat(d) = new DenseMatrix(M, M, arrTransmat2.slice(d * multM, (d + 1) * multM)))

        (0 until D).foreach(d =>
          (0 until M).foreach(i => transmat(d)(i, ::) := normalize(transmat(d)(i, ::).t, 1.0).t))

        prior = normalize(new DenseVector(newvalues.getAs[Seq[Double]](1).toArray), 1.0)
        obsmat = Utils.mkstochastic(new DenseMatrix(M, k, newvalues.getAs[Seq[Double]](3).toArray))

        arrTransmat2 = Array()
        (0 until D).foreach(d => arrTransmat2 = arrTransmat2 ++ transmat(d).toArray)

        nhsmm.Utils.writeresult(path_Class_baumwelch + kfold,
          kfold + ";" +
            it + ";" +
            M + ";" +
            k + ";" +
            D + ";" +
            prior.toArray.mkString(",") + ";" +
            //validar si lo genera correctamente
            arrTransmat2.mkString(",") + ";" +
            obsmat.toArray.mkString(",") + ";" +
            loglik + "\n")

        if (Utils.emconverged(loglik, antloglik, epsilon)) {
          log.info("End Iteration: " + it)
          log.info("-----------------------------------------------------------------------------------------")
          break
        }
        antloglik = loglik

        obstrained.unpersist()
        obstrained = observations
          .withColumn("M", lit(M))
          .withColumn("k", lit(k))
          .withColumn("D", lit(D))
          .withColumn("Pi", lit(prior.toArray))
          .withColumn("A", lit(arrTransmat2))
          .withColumn("B", lit(obsmat.toArray))
          .withColumn("obs", udf_toarray(col("str_obs")))
          .withColumn("T", udf_obssize(col("obs")))
        log.info("End Iteration: " + it)
        log.info("-----------------------------------------------------------------------------------------")
      })
    }
    (prior, transmat, obsmat)
  }

  def validate(observations: DataFrame, M: Int, k: Int, D: Int,
               initialPi: DenseVector[Double], initialA: DenseVector[DenseMatrix[Double]], initialB: DenseMatrix[Double]):
  DataFrame = {

    var arrTransmat: Array[Double] = Array()
    (0 until D).foreach(d => arrTransmat = arrTransmat ++ initialA(d).toArray)

    observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("D", lit(D))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(arrTransmat))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))
      .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
      .withColumn("prob", udf_fwd(col("M"), col("D"), col("T"), col("Pi"), col("A"), col("obslik")))
      .drop("str_obs", "M", "k", "D", "Pi", "A", "B", "P", "obs", "T", "obslik")
  }

  /** * udf functions ****/
  val udf_toarray: UserDefinedFunction = udf((s: String) => s.split(";").map(_.toInt))
  val udf_obssize: UserDefinedFunction = udf((s: Seq[Int]) => s.length)

  /** * udf_multinomialprob ****/
  val udf_multinomialprob: UserDefinedFunction = udf((obs: Seq[Int], M: Int, k: Int, T: Int, B: Seq[Double]) => {
    val funB: DenseMatrix[Double] = new DenseMatrix(M, k, B.toArray)
    val output: DenseMatrix[Double] = DenseMatrix.tabulate(M, T) { case (m, t) => funB(m, obs(t)) }
    output.toArray
  })

  val udf_fwdback: UserDefinedFunction = udf((M: Int, k: Int, D: Int, T: Int, Pi: Seq[Double], A: Seq[Double], obslik: Seq[Double], obs: Seq[Int]) => {

    val funA: DenseVector[DenseMatrix[Double]] = DenseVector.fill(D) {
      DenseMatrix.zeros[Double](M, M)
    }

    val multM = M * M
    (0 until D).foreach(d => funA(d) = new DenseMatrix(M, M, A.toArray.slice(d * multM, (d + 1) * multM)))

    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    /**
      * Yu, S.-Z. (2016). Hidden Semi-Markov Models Theory, Algorithms and Applications. In Hidden Semi-Markov Models (pp. 1–26). Elsevier. https://doi.org/10.1016/B978-0-12-802767-7.00001-2
      * Algorithm -> 5.2
      */

    /**
      * Forwards variables
      */
    val scale: DenseVector[Double] = DenseVector.ones[Double](2 * T * M * D)
    var scalaindex = 0

    val alpha = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    (0 until M).foreach(j => alpha(0)(j, 0) = funPi(j) * funObslik(j, 0))
    alpha(0)(::, 0) := Utils.normalise(alpha(0)(::, 0), scale, 0)

    (1 until T).foreach(t => {
      (0 until M).foreach(j => {

        (0 until M).foreach(i => if (i != j)
          alpha(t)(j, 0) = alpha(t)(j, 0) + (alpha(t - 1)(i, 0) * funA(0)(i, j) * funObslik(j, t)))

        (1 until D).foreach(d => {

          var tempAlpha = 0.0
          (0 until M).foreach(i => if (i != j) tempAlpha = tempAlpha + (alpha(t - 1)(i, d) * funA(d)(i, j) * funObslik(j, t)))
          alpha(t)(j, 0) = alpha(t)(j, 0) + tempAlpha

          alpha(t)(::, 0) := Utils.normalise(alpha(t)(::, 0), scale, scalaindex)
          scalaindex = scalaindex + 1

          alpha(t)(j, d) = alpha(t - 1)(j, d - 1) * funA(d - 1)(j, j) * funObslik(j, t)
          //normalización

          alpha(t)(::, d) := Utils.normalise(alpha(t)(::, d), scale, scalaindex)
          scalaindex = scalaindex + 1

        })
      })
    })

    val loglik: Double = sum(scale.map(Math.log))

    /**
      * Backwards variables
      */
    val beta = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    (0 until M).foreach(j => (0 until D).foreach(d => beta(T - 1)(j, d) = 1.0))

    for (t <- T - 2 to 0 by -1) {
      (0 until M).foreach(j => {
        //probar si es "D - 1" o "D", see algorithm
        (0 until D - 1).foreach(d => {

          (0 until M).foreach(i => if (i != j)
            beta(t)(j, d) = beta(t)(j, d) +
              (funA(d)(j, i) * beta(t + 1)(i, 0) * funObslik(i, t + 1)) +
                (funA(d)(j, j) * beta(t + 1)(j, d + 1) * funObslik(j, t + 1)))
          beta(t)(::, d) :=  normalize(beta(t)(::, d), 1.0)
        })
        //beta(t)(j, ::) :=  normalize(beta(t)(j, ::).t, 1.0).t
      })
    }

    /**
      * Yu, S.-Z. (2016). Hidden Semi-Markov Models Theory, Algorithms and Applications. In Hidden Semi-Markov Models (pp. 1–26). Elsevier. https://doi.org/10.1016/B978-0-12-802767-7.00001-2
      * Section -> 5.2.1
      */
    val matrixi = DenseMatrix.fill(T, D) {
      DenseMatrix.zeros[Double](M, M)
    }

    //check index of T
    (0 until T - 1).foreach(t => {
      (0 until M).foreach(i => {
        (0 until D).foreach(d => {
          (0 until M).foreach(j => {
            if (i != j)
              matrixi(t, d)(i, j) = alpha(t)(i, d) * funA(d)(i, j) * funObslik(j, t + 1) * beta(t + 1)(j, 0)
            else if (d <= D - 2)
              matrixi(t, d)(i, j) = alpha(t)(i, d) * funA(d)(i, i) * funObslik(i, t + 1) * beta(t + 1)(i, d + 1)
          })
          matrixi(t, d)(i, ::) := Utils.normalise(matrixi(t, d)(i, ::).t).t
        })
      })
    })

    /**
      * Yu, S.-Z. (2016). Hidden Semi-Markov Models Theory, Algorithms and Applications. In Hidden Semi-Markov Models (pp. 1–26). Elsevier. https://doi.org/10.1016/B978-0-12-802767-7.00001-2
      * Section -> 5.2.1
      */
    val matrixn = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    (0 until T).foreach(t => {
      (0 until M).foreach(i => {
        (0 until D).foreach(d => {
          (0 until M).foreach(j => if (i != j) matrixn(t)(i, d) = matrixn(t)(i, d) + matrixi(t, d)(i, j))
        })
        matrixn(t)(i, ::) := normalize(matrixn(t)(i, ::).t, 1.0).t
      })
      //matrixn(t) := Utils.normalise(matrixn(t))
    })

    /**
      * Yu, S.-Z. (2016). Hidden Semi-Markov Models Theory, Algorithms and Applications. In Hidden Semi-Markov Models (pp. 1–26). Elsevier. https://doi.org/10.1016/B978-0-12-802767-7.00001-2
      * Equation -> 2.13
      */
    val matrixg = DenseMatrix.zeros[Double](M, T)

    (0 until T).foreach(t => {
      (0 until M).foreach(j => {
        (t until T).foreach(tao => {
          (tao - t until D).foreach(d => {
            matrixg(j, t) = matrixg(j, t) + matrixn(tao)(j, d)
          })
        })
      })
      matrixg(::, t) := Utils.normalise(matrixg(::, t))
    })

    /**
      * Yu, S.-Z. (2016). Hidden Semi-Markov Models Theory, Algorithms and Applications. In Hidden Semi-Markov Models (pp. 1–26). Elsevier. https://doi.org/10.1016/B978-0-12-802767-7.00001-2
      * Section -> 5.2.1
      */
    val newPi: DenseVector[Double] = new DenseVector(matrixg(::, 0).toArray)
    newPi := Utils.normalise(newPi)

    val newA: DenseVector[DenseMatrix[Double]] = DenseVector.fill(D) {
      DenseMatrix.zeros[Double](M, M)
    }

    (0 until D).foreach(d => {
      (0 until M).foreach(i => {
        (0 until M).foreach(j => {
          (0 until T).foreach(t => {
            newA(d)(i, j) = newA(d)(i, j) + matrixi(t, d)(i, j)
          })
        })
        newA(d)(i, ::) := Utils.normalise(newA(d)(i, ::).t).t
      })
    })

    val newB = DenseMatrix.zeros[Double](M, k)

    (0 until M).foreach(j => {
      (0 until k).foreach(v => {
        (0 until T).foreach(t => {
          if (obs(t) == v) {
            newB(j, v) = newB(j, v) + matrixg(j, t)
          }
        })
      })
      newB(j, ::) := Utils.normalise(newB(j, ::).t).t
    })

    var arrTransmat: Array[Double] = Array()
    (0 until D).foreach(d => arrTransmat = arrTransmat ++ newA(d).toArray)

    (loglik, newPi.toArray, arrTransmat, newB.toArray)
  })

  val udf_loglik: UserDefinedFunction = udf((input: Row) => input.get(0).asInstanceOf[Double])
  val udf_newPi: UserDefinedFunction = udf((input: Row) => input.get(1).asInstanceOf[Seq[Double]])
  val udf_newA: UserDefinedFunction = udf((input: Row) => input.get(2).asInstanceOf[Seq[Double]])
  val udf_newB: UserDefinedFunction = udf((input: Row) => input.get(3).asInstanceOf[Seq[Double]])

  /** * Por optimizar ****/
  val udf_fwd: UserDefinedFunction = udf((M: Int, D: Int, T: Int, Pi: Seq[Double], A: Seq[Double], obslik: Seq[Double]) => {

    val funA: DenseVector[DenseMatrix[Double]] = DenseVector.fill(D) {
      DenseMatrix.zeros[Double](M, M)
    }

    val multM = M * M
    (0 until D).foreach(d => funA(d) = new DenseMatrix(M, M, A.toArray.slice(d * multM, (d + 1) * multM)))

    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    /**
      * Yu, S.-Z. (2016). Hidden Semi-Markov Models Theory, Algorithms and Applications. In Hidden Semi-Markov Models (pp. 1–26). Elsevier. https://doi.org/10.1016/B978-0-12-802767-7.00001-2
      * Algorithm -> 5.2
      */

    /**
      * Forwards variables
      */
    val scale: DenseVector[Double] = DenseVector.ones[Double](T * D)
    var scalaindex = 1

    val alpha = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    (0 until M).foreach(j => alpha(0)(j, 0) = funPi(j) * funObslik(j, 0))
    alpha(0)(::, 0) := Utils.normalise(alpha(0)(::, 0), scale, 0)

    (1 until T).foreach(t => {
      (0 until M).foreach(j => {

        (0 until M).foreach(i => if (i != j)
          alpha(t)(j, 0) = alpha(t)(j, 0) + (alpha(t - 1)(i, 0) * funA(0)(i, j) * funObslik(j, t)))

        (1 until D).foreach(d => {

          var tempAlpha = 0.0
          (0 until M).foreach(i => if (i != j) tempAlpha = tempAlpha + (alpha(t - 1)(i, d) * funA(d)(i, j) * funObslik(j, t)))
          alpha(t)(j, 0) = alpha(t)(j, 0) + tempAlpha

          alpha(t)(::, 0) := Utils.normalise(alpha(t)(::, 0), scale, scalaindex)
          scalaindex = scalaindex + 1

          alpha(t)(j, d) = alpha(t - 1)(j, d - 1) * funA(d - 1)(j, j) * funObslik(j, t)
          //normalización

          alpha(t)(::, d) := Utils.normalise(alpha(t)(::, d), scale, scalaindex)
          scalaindex = scalaindex + 1

        })
      })
    })

    val loglik: Double = sum(scale.map(Math.log))
    loglik
  })


}
