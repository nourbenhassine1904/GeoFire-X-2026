"""Real archive metadata at full index scale; raster QC coverage stays explicit."""
import json
import time
from pathlib import Path

from geofirex.cloud_corpus import validate_index, seal
from geofirex.corpus import json_new, sha256_file


def run(index_path, qc_path, destination):
    from pyspark.sql import SparkSession, functions as F
    index = json.loads(Path(index_path).read_text())
    validate_index(index, official=True)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    spark = (SparkSession.builder.master('local[2]').appName('GeoFireX-official-metadata')
        .config('spark.ui.enabled', 'false').config('spark.sql.shuffle.partitions', '4').getOrCreate())
    spark.sparkContext.setLogLevel('WARN')
    try:
        ingestion = time.perf_counter()
        columns = ['event_id', 'date', 'year', 'path', 'compressed_bytes', 'geotiff_bytes', 'crc32']
        frame = spark.createDataFrame([{k: entry[k] for k in columns} for entry in index['entries']]).cache()
        count = frame.count()
        events = frame.select('event_id').distinct().count()
        duplicates = frame.groupBy('event_id', 'date').count().filter('count > 1').count()
        # An event can legitimately continue into the following calendar year.
        cross_year = frame.filter(F.substring('date', 1, 4).cast('int') != F.col('year')).count()
        invalid = frame.filter((F.col('compressed_bytes') <= 0) | (F.col('geotiff_bytes') <= 0)
            | (F.substring('event_id', 1, 4).cast('int') != F.col('year'))
            | F.to_date('date').isNull()).count()
        assert (count, events, duplicates, invalid) == (13607, 607, 0, 0)
        ingestion_seconds = time.perf_counter()-ingestion
        frame.write.mode('errorifexists').partitionBy('year').parquet(str(out/'index-parquet'))
        round_trip = spark.read.parquet(str(out/'index-parquet'))
        assert round_trip.count() == count
        qc = spark.read.parquet(str(qc_path))
        qc_count = qc.count()
        assert qc.groupBy('event_id', 'date').count().filter('count > 1').count() == 0
        unmatched = qc.join(frame, ['event_id', 'date'], 'left_anti').count()
        assert unmatched == 0
        summary = (frame.groupBy('year').agg(F.count('*').alias('observations'),
            F.countDistinct('event_id').alias('events'), F.sum('geotiff_bytes').alias('indexed_bytes'))
            .orderBy('year'))
        annual = [r.asDict() for r in summary.collect()]
        summary.write.mode('errorifexists').parquet(str(out/'annual-parquet'))
        qc.groupBy('year', 'geo_partition').agg(F.count('*').alias('observations'),
            F.countDistinct('event_id').alias('events'), F.sum('fire_pixels').alias('fire_pixels'))\
            .write.mode('errorifexists').parquet(str(out/'raster-qc-parquet'))
        report = seal({'status': 'SPARK_EXECUTED', 'spark_version': spark.version, 'master': 'local[2]',
            'scope': '607-event official archive index; existing local raster QC covers only 108 observations / 5 events',
            'raster_corpus_acquired': False, 'distributed_cluster': False,
            'index_observations': count, 'index_events': events, 'duplicate_keys': duplicates, 'invalid_rows': invalid,
            'cross_calendar_year_observations': cross_year,
            'raster_qc_observations': qc_count, 'raster_qc_events': qc.select('event_id').distinct().count(),
            'unmatched_raster_qc_keys': unmatched, 'index_ingestion_qc_seconds': ingestion_seconds,
            'index_rows_per_second_including_qc': count/ingestion_seconds,
            'total_seconds': time.perf_counter()-started, 'annual': annual,
            'source_index_sha256': sha256_file(index_path), 'raster_qc_parquet_sha256': sha256_file(qc_path)})
        json_new(out/'spark-report.json', report)
        return report
    finally:
        spark.stop()
