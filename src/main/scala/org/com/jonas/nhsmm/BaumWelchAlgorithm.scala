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
      prior = new DenseVector(arraymodel(4).split(",").map(_.toDouble))

      arrTransmat = arraymodel(5).split(",").map(_.toDouble)

      transmat = DenseVector.fill(D) {
        DenseMatrix.zeros[Double](M, M)
      }

      (0 until D).foreach(d => transmat(d) = new DenseMatrix(M, M, arrTransmat.slice((d * M * M), (((d + 1) * (M * M)) - 1))))

      obsmat = new DenseMatrix(M, k, arraymodel(6).split(",").map(_.toDouble))
      antloglik = arraymodel(8).toDouble
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
      (0 until maxIterations).foreach(it => {
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

        prior = normalize(new DenseVector(newvalues.getAs[Seq[Double]](1).toArray), 1.0)

        var arrTransmat2: Array[Double] = newvalues.getAs[Seq[Double]](2).toArray
        (0 until D).foreach(d => transmat(d) = Utils.mkstochastic(new DenseMatrix(M, M, arrTransmat2.slice((d * M * M), (((d + 1) * (M * M)) - 1)))))

        obsmat = Utils.mkstochastic(new DenseMatrix(M, k, newvalues.getAs[Seq[Double]](3).toArray))

        nhsmm.Utils.writeresult(path_Class_baumwelch + kfold,
          kfold + ";" +
            it + ";" +
            M + ";" +
            k + ";" +
            D + ";" +
            prior.toArray.mkString(",") + ";" +
            transmat.toArray.mkString(",") + ";" +
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
          .withColumn("A", lit(transmat.toArray))
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
               initialPi: DenseVector[Double], initialA: DenseVector[DenseMatrix[Double]], initialB: DenseMatrix[Double], initialP: DenseMatrix[Double]):
  DataFrame = {
    observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("D", lit(D))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("P", lit(initialP.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))
      .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
      .withColumn("prob", udf_fwd(col("M"), col("D"), col("T"), col("Pi"), col("A"), col("P"), col("obslik")))
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

  /** * udf_multinomialprob "optimized" ****/
  val udf_multinomialprob2: UserDefinedFunction = udf((obs: Seq[Int], M: Int, k: Int, T: Int, B: Seq[Double]) => {
    val output = Array.empty[Double]
    (0 until T).foreach(j => {
      val Mj = M * j
      (0 until M).foreach(i => output :+ B(Mj + i))
    })
    output
  })

  val udf_fwdback: UserDefinedFunction = udf((M: Int, k: Int, D: Int, T: Int, Pi: Seq[Double], A: Seq[Double], obslik: Seq[Double], obs: Seq[Int]) => {

    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA: DenseVector[DenseMatrix[Double]] = DenseVector.fill(D) {
      DenseMatrix.zeros[Double](M, M)
    }

    (0 until D).foreach(d => funA(d) = new DenseMatrix(M, M, A.toArray.slice((d * M * M), (((d + 1) * (M * M)) - 1))))

    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    val scale: DenseVector[Double] = DenseVector.ones[Double](T * D)
    val alpha = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    (0 until M).foreach(j => alpha(0)(j, 0) = funPi(j) * funObslik(j, 0))
    alpha(0)(::, 0) := Utils.normalise(alpha(0)(::, 0), scale, 0)

    var scalaindex = 1

    //se cambio el orden de los ciclos para realizar normalización en j
    //probar normalizar por matrices
    /** *
      * IMPORTANTE probar normalizando alpha por fila, y normalizando beta por matrix
      *
      */
    (1 until T).foreach(t => {

      (1 until D).foreach(d => {

        (0 until M).foreach(j => {

          (0 until M).foreach(i => alpha(t)(j, 0) = alpha(t - 1)(i, 0) * funA(0)(i, j) * funObslik(j, t))


          var temp = 0.0
          (0 until M).foreach(i => temp = temp + alpha(t - 1)(i, d) * funA(d)(i, j) * funObslik(j, t))

          alpha(t)(j, 0) = alpha(t)(j, 0) + temp
          alpha(t)(j, d) = alpha(t - 1)(j, d - 1) * funA(d - 1)(j, j) * funObslik(j, t)
        })
        alpha(t)(::, d) := Utils.normalise(alpha(t)(::, d), scale, scalaindex)
        scalaindex = scalaindex + 1
      })
    })

    val loglik: Double = sum(scale.map(Math.log))

    val beta = DenseVector.fill(T) {
      DenseMatrix.ones[Double](M, D)
    }

    //es necesio normalizart la fila de unos?
    for (t <- T - 2 to 0 by -1) {
      (0 until D).foreach(d => {
        (0 until M).foreach(j => {
          beta(t)(j, d) = 0.0
          (0 until M).foreach(i => beta(t)(j, d) = beta(t)(j, d) + (funA(d)(j, i) * beta(t + 1)(i, 0) * funObslik(i, t + 1)) + (funA(d)(j, j) * beta(t + 1)(j, d + 1) * funObslik(j, t + 1)))
        })
        beta(t)(::, d) := normalize(beta(t)(::, d), 1.0)
      })
    }

    val matrixi = DenseMatrix.fill(T, D) {
      DenseMatrix.zeros[Double](M, M)
    }

    (0 until T).foreach(t => {
      (0 until D).foreach(d => {
        (0 until M).foreach(i => {
          (0 until M).foreach(j => {
            if (i != j) {
              matrixi(t, d)(i, j) = alpha(t)(i, d) * funA(d)(i, j) * funObslik(j, t + 1) * beta(t + 1)(j, 1)
            } else if (d <= D - 2) {
              matrixi(t, d)(i, j) = alpha(t)(i, d) * funA(d)(i, i) * funObslik(i, t + 1) * beta(t + 1)(i, d + 1)
            }
          })
        })
        matrixi(t, d) := Utils.normalise(matrixi(t, d))
      })
    })

    val matrixn = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    (0 until T).foreach(t => {
      (0 until D).foreach(d => {
        (0 until M).foreach(i => {
          (0 until M).foreach(j => matrixn(t)(i, d) = matrixn(t)(i, d) + matrixi(t, d)(i, j))
        })
        matrixn(t)(::, d) := normalize(matrixn(t)(::, d), 1.0)
      })
    })

    val matrixg = DenseMatrix.zeros[Double](T, M)

    (0 until T).foreach(t => {
      (0 until M).foreach(j => {
        (t until T).foreach(tao => {
          (tao - t + 1 until D).foreach(d => {
            matrixg(t, j) = matrixg(t, j) + matrixn(tao)(j, d)
          })
        })
      })
      matrixg(t, ::) := normalize(matrixg(t, ::), 1.0)
    })

    val newPi: DenseVector[Double] = new DenseVector(matrixg(0, ::).t.toArray)

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
        newA(d)(i, ::) := normalize(newA(d)(i, ::), 1.0)
      })
    })

    val newB = DenseMatrix.zeros[Double](M, k)

    (0 until M).foreach(j => {
      (0 until k).foreach(v => {
        (0 until T).foreach(t => {
          if (obs(t) == v) {
            newB(j, v) = newB(j, v) + matrixg(t, j)
          }
        })
      })
      newB(j, ::) := normalize(newB(j, ::).t, 1.0).t
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
  val udf_fwd: UserDefinedFunction = udf((M: Int, D: Int, T: Int, Pi: Seq[Double], A: Seq[Double], P: Seq[Double], obslik: Seq[Double]) => {

    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA: DenseMatrix[Double] = new DenseMatrix(M, M, A.toArray)
    val funP: DenseMatrix[Double] = new DenseMatrix(M, D, P.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    /**
      * Matriz u(t,j,d)
      */
    val matrixu: DenseVector[DenseMatrix[Double]] = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    /** * optimizar unificando **/
    (0 until T).foreach(t =>
      (0 until M).foreach(j => matrixu(t)(j, 0) = funObslik(j, t)))

    (0 until T).foreach(t =>
      (0 until M).foreach(j =>
        (1 until D).foreach(d =>
          if (d - 1 > -1 && t - d + 1 > -1)
            matrixu(t)(j, d) = matrixu(t)(j, d - 1) * funObslik(j, t - d + 1))))

    /**
      * Forwards variables
      */
    val scale: DenseVector[Double] = DenseVector.ones[Double](T)
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    val alphaprime: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T + 1)

    alphaprime(::, 0) := normalize(funPi, 1.0)
    (0 until T).foreach(t => {
      (0 until M).foreach(j =>
        (0 until D).foreach(d =>
          if (t - d + 1 > -1 && t - d + 1 < T + 1)
            alpha(j, t) = alpha(j, t) + (alphaprime(j, t - d + 1) * funP(j, d) * matrixu(t)(j, d))))
      alpha(::, t) := Utils.normalise(alpha(::, t), scale, t)
      (0 until M).foreach(j =>
        (0 until M).foreach(i => alphaprime(j, t + 1) = alphaprime(j, t + 1) + alpha(i, t) * funA(i, j)))
      alphaprime(::, t + 1) := normalize(alphaprime(::, t + 1), 1.0)
    })

    //var loglik: Double = 0.0
    //if (scale.toArray.filter(i => i == 0).isEmpty) loglik = sum(scale.map(Math.log)) else loglik = Double.NegativeInfinity
    //if (scale.findAll(i => i == 0).isEmpty) loglik = sum(scale.map(Math.log)) else loglik = Double.NegativeInfinity
    //Modificación validar si funciona
    val loglik: Double = sum(scale.map(Math.log))
    loglik
  })


}
