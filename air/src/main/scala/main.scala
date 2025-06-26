import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

object PrepareAirQualityData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AirQualityHealthPreparation")
      .master("local[*]")
      .getOrCreate()

    val dataPath = "src/main/scala/air_quality_health_dataset.csv"
    val outputCsvPath = "src/main/scala/aq_health_clean"

    // Lecture des données
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(dataPath)

    println("Colonnes originales:")
    df.columns.foreach(println)
    println(s"Nombre de lignes originales: ${df.count()}")

    // Identification et affichage des valeurs uniques de population_density
    println("\n=== ANALYSE DE POPULATION_DENSITY ===")
    val uniquePopDensity = df.select("population_density")
      .distinct()
      .collect()
      .map(_.getString(0))
      .filter(_ != null)
      .sorted

    println("Valeurs uniques trouvées pour population_density:")
    uniquePopDensity.zipWithIndex.foreach { case (value, index) =>
      println(s"$index -> $value")
    }

    // Création du mapping
    val popDensityMapping = uniquePopDensity.zipWithIndex.toMap
    println(s"\nMapping créé: $popDensityMapping")

    val dfCasted = df
      .withColumn("pm2_5", col("pm2_5").cast("double"))
      .withColumn("pm10", col("pm10").cast("double"))
      .withColumn("no2", col("no2").cast("double"))
      .withColumn("o3", col("o3").cast("double"))
      .withColumn("temperature", col("temperature").cast("double"))
      .withColumn("humidity", col("humidity").cast("double"))
      .withColumn("aqi", col("aqi").cast("double"))
      .withColumn("hospital_capacity", col("hospital_capacity").cast("double"))
      .withColumn("hospital_admissions", col("hospital_admissions").cast("double"))

    // Application du mapping sur population_density
    val dfWithMappedPopDensity = popDensityMapping.foldLeft(dfCasted) { case (currentDf, (value, index)) =>
      currentDf.withColumn("population_density",
        when(col("population_density") === value, lit(index))
          .otherwise(col("population_density"))
      )
    }.withColumn("population_density", col("population_density").cast("int"))

    println("\n=== VÉRIFICATION DU MAPPING ===")
    println("Distribution après mapping:")
    dfWithMappedPopDensity.groupBy("population_density").count().orderBy("population_density").show()

    val validYears = dfWithMappedPopDensity
      .filter(year(col("date")) <= 2025)
      .select(year(col("date")).alias("year"))
      .distinct()
      .collect()
      .map(_.getInt(0))
      .sorted

    val firstValidYear = if (validYears.nonEmpty) validYears.head else 2020
    val replacementYear = firstValidYear - 1

    println(s"Première année valide trouvée: $firstValidYear")
    println(s"Les années > 2025 seront remplacées par: $replacementYear")

    // Correction des dates
    val dfCorrectedDates = dfWithMappedPopDensity.withColumn("date",
      when(year(col("date")) > 2025,
        date_format(
          to_date(
            concat(lit(replacementYear), lit("-"),
              month(col("date")), lit("-"),
              dayofmonth(col("date"))),
            "yyyy-M-d"
          ),
          "yyyy-MM-dd"
        )
      ).otherwise(col("date"))
    )

    // Affichage des corrections effectuées
    val correctedCount = dfWithMappedPopDensity.filter(year(col("date")) > 2025).count()
    println(s"Nombre de dates corrigées: $correctedCount")

    // Suppression des lignes avec hospital_admissions manquant (variable cible)
    val dfWithoutNullTarget = dfCorrectedDates.na.drop(cols = Seq("hospital_admissions"))

    // Imputation des valeurs manquantes par la MÉDIANE pour les colonnes numériques
    val numericCols = Seq("pm2_5","pm10","no2","o3","temperature","humidity","aqi","hospital_capacity","population_density")

    println("Imputation par la médiane en cours...")

    val medians = numericCols.map { colName =>
      val median = dfWithoutNullTarget.stat.approxQuantile(colName, Array(0.5), 0.0)(0)
      println(s"Médiane pour $colName: $median")
      (colName, median)
    }.toMap

    val dfImputed = numericCols.foldLeft(dfWithoutNullTarget) { (df, colName) =>
      df.withColumn(colName,
        when(col(colName).isNull, lit(medians(colName)))
          .otherwise(col(colName))
      )
    }

    // Suppression des outliers sur hospital_admissions (variable cible)
    val quantiles = dfImputed.stat.approxQuantile("hospital_admissions", Array(0.25, 0.75), 0.0)
    val q1 = quantiles(0)
    val q3 = quantiles(1)
    val iqr = q3 - q1
    val lower = q1 - 1.5 * iqr
    val upper = q3 + 1.5 * iqr

    println(s"Suppression des outliers: hospital_admissions < $lower ou > $upper")

    val finalDF = dfImputed.filter(col("hospital_admissions").between(lower, upper))

    println(s"Nombre de lignes après traitement: ${finalDF.count()}")
    println("Colonnes finales:")
    finalDF.columns.foreach(println)

    println("\nÉchantillon des données finales:")
    finalDF.select("city", "population_density", "hospital_admissions").show(10)

    println("\nÉchantillon des dates après correction:")
    finalDF.select("date").distinct().orderBy("date").show(10)

    import java.io.File
    def deleteDirectory(dir: File): Unit = {
      if (dir.exists()) {
        dir.listFiles().foreach { file =>
          if (file.isDirectory) deleteDirectory(file)
          else file.delete()
        }
        dir.delete()
      }
    }
    deleteDirectory(new File(outputCsvPath))

    finalDF.coalesce(1).write
      .mode("overwrite")
      .option("header", "true")
      .csv(outputCsvPath)

    println(s"Données sauvegardées dans: $outputCsvPath")
    println("Le fichier CSV se trouve dans le dossier créé par Spark")

    println("\n=== MAPPING FINAL POPULATION_DENSITY ===")
    uniquePopDensity.zipWithIndex.foreach { case (value, index) =>
      println(s"$value -> $index")
    }

    spark.stop()
  }
}