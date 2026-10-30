"""Future distributed metadata aggregation; requires a separately provisioned Spark.

Does not read GeoTIFF arrays. A local[*] run is a local pilot, not a cluster run.
"""
import argparse


def aggregate_metadata(spark,input_path,output_path):
    from pyspark.sql import functions as F
    observations=spark.read.parquet(input_path)
    duplicate=observations.groupBy("event_id","date").count().filter("count > 1").limit(1).count()
    if duplicate:
        raise ValueError("Duplicate observation keys")
    (observations.groupBy("year","geo_partition")
     .agg(F.count("*").alias("observations"),F.countDistinct("event_id").alias("events"),
          F.sum("fire_pixels").alias("fire_pixels"),F.sum("pixel_count").alias("pixels"))
     .write.mode("errorifexists").partitionBy("year").parquet(output_path))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--master",default="local[2]")
    args=p.parse_args()
    from pyspark.sql import SparkSession
    spark=SparkSession.builder.master(args.master).appName("GeoFireX-metadata-QC").getOrCreate()
    try:
        aggregate_metadata(spark,args.input,args.output)
    finally:
        spark.stop()
