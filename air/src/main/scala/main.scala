import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

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

    // Suppression des lignes avec hospital_admissions manquant (variable cible)
    val dfWithoutNullTarget = dfCasted.na.drop(cols = Seq("hospital_admissions"))

    // Imputation des valeurs manquantes par la moyenne pour les colonnes numériques
    val numericCols = Seq("pm2_5","pm10","no2","o3","temperature","humidity","aqi","hospital_capacity")
    val imputer = new Imputer()
      .setStrategy("mean")
      .setInputCols(numericCols.toArray)
      .setOutputCols(numericCols.toArray)

    val dfImputed = imputer.fit(dfWithoutNullTarget).transform(dfWithoutNullTarget)

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
    println("Colonnes finales (identiques aux originales):")
    finalDF.columns.foreach(println)

    // Suppression manuelle du dossier de sortie si il existe
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

    // Sauvegarde en CSV avec les mêmes colonnes que l'original
    finalDF.coalesce(1).write
      .mode("overwrite")
      .option("header", "true")
      .csv(outputCsvPath)

    println(s"Données sauvegardées dans: $outputCsvPath")
    println("Le fichier CSV se trouve dans le dossier créé par Spark")
    spark.stop()
  }
}