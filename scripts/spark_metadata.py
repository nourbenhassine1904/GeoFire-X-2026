"""Future distributed metadata aggregation; requires a separately provisioned Spark.

Does not read GeoTIFF arrays. A local[*] run is a local pilot, not a cluster run.
"""
import argparse


def aggregate_metadata(spark,input_path,output_path, *, min_events=0, min_observations=0):
    from pyspark.sql import functions as F
    observations=spark.read.parquet(input_path)
    count = observations.count()
    events = observations.select("event_id").distinct().count()
    if count < min_observations or events < min_events:
        raise ValueError("Metadata input is below the declared representative scale")
    duplicate=observations.groupBy("event_id","date").count().filter("count > 1").limit(1).count()
    if duplicate:
        raise ValueError("Duplicate observation keys")
    (observations.groupBy("year","geo_partition")
     .agg(F.count("*").alias("observations"),F.countDistinct("event_id").alias("events"),
          F.sum("fire_pixels").alias("fire_pixels"),F.sum("pixel_count").alias("pixels"))
     .write.mode("errorifexists").partitionBy("year").parquet(output_path))
    return {"observations": count, "events": events, "processing": "Spark metadata aggregation; no GPU/raster distribution claim"}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--master",default="local[2]")
    p.add_argument("--min-events", type=int, default=30)
    p.add_argument("--min-observations", type=int, default=500)
    p.add_argument("--out-report", required=True)
    args=p.parse_args()
    from pyspark.sql import SparkSession
    spark=SparkSession.builder.master(args.master).appName("GeoFireX-metadata-QC").getOrCreate()
    try:
        from geofirex.corpus import json_new
        report = aggregate_metadata(spark,args.input,args.output,min_events=args.min_events,min_observations=args.min_observations)
        json_new(args.out_report, {**report, "master": args.master})
    finally:
        spark.stop()
