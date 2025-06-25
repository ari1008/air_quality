import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._

object PrepareAirQualityData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AirQualityHealthPreparation")
      .config("spark.driver.host", "127.0.0.1")
      .master("local[*]")
      .getOrCreate()

    val dataPath = "/Users/aristidefumo/Documents/school/distribue/air_quality/dataset/air_quality_health_dataset.csv"
    val outputParquetPath = "/Users/aristidefumo/Documents/school/distribue/air_quality/dataset/air_quality_health_dataset.csvaq_health_clean.parquet"

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(dataPath)

    val df2 = df
      .withColumn("date_ts", to_timestamp(col("date"), "yyyy-MM-dd"))
      .withColumn("year", year(col("date_ts")))
      .withColumn("month", month(col("date_ts")))
      .withColumn("dayofyear", dayofyear(col("date_ts")))
      .drop("date")
      .withColumn("hospital_admissions", col("hospital_admissions").cast("double"))
      .withColumn("hospital_capacity", col("hospital_capacity").cast("double"))
      .withColumn("aqi", col("aqi").cast("double"))
      .withColumn("humidity", col("humidity").cast("double"))

    val df3 = df2.na.drop(cols = Seq("hospital_admissions"))

    val numericCols = Seq("pm2_5", "pm10", "no2", "o3", "temperature", "humidity", "aqi", "hospital_capacity")
    val imputer = new Imputer()
      .setStrategy("mean")
      .setInputCols(numericCols.toArray)
      .setOutputCols(numericCols.toArray)

    val quantiles = df3.stat.approxQuantile("hospital_admissions", Array(0.25, 0.75), 0.0)
    val (q1, q3) = (quantiles(0), quantiles(1))
    val iqr = q3 - q1
    val (lower, upper) = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    val dfFiltered = df3.filter(col("hospital_admissions").between(lower, upper))

    val indexers = Seq("city", "population_density").map { c =>
      new StringIndexer().setInputCol(c).setOutputCol(s"${c}_idx").setHandleInvalid("keep")
    }

    val encoders = Seq("city_idx", "population_density_idx").map { c =>
      new OneHotEncoder().setInputCol(c).setOutputCol(s"${c}_vec").setDropLast(false)
    }

    val assembler = new VectorAssembler()
      .setInputCols((numericCols ++ Seq("year", "month", "dayofyear", "city_idx_vec", "population_density_idx_vec")).toArray)
      .setOutputCol("features")

    val stages = Seq(imputer) ++ indexers ++ encoders ++ Seq(assembler)
    val pipeline = new Pipeline().setStages(stages.toArray)
    val model = pipeline.fit(dfFiltered)
    val finalDF = model.transform(dfFiltered)
      .select("features", "hospital_admissions")

    finalDF.write.mode("overwrite").parquet(outputParquetPath)

    println(s"✅ Données sauvegardées dans $outputParquetPath")
    spark.stop()
  }
}
