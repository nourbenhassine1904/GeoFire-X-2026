"""Actual local Spark SQL demo over existing certified metadata; no raster audit."""
import argparse
import json
from pathlib import Path
import time

from geofirex.mission_streaming import save,sha

def run(prepared,out):
    from pyspark.sql import SparkSession,functions as F
    import pyarrow as pa
    import pyarrow.parquet as pq
    out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    spark=(SparkSession.builder.master('local[2]').appName('GeoFireX-mission-20261111')
        .config('spark.ui.enabled','false').config('spark.sql.shuffle.partitions','2')
        .config('spark.driver.memory','512m').config('spark.executor.memory','512m')
        .config('spark.driver.extraJavaOptions','-XX:MaxDirectMemorySize=128m -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m').getOrCreate())
    spark.sparkContext.setLogLevel('WARN');init=time.perf_counter()-started
    try:
        t=time.perf_counter();obs=spark.read.parquet(str(Path(prepared)/'observations.parquet'));obs.createOrReplaceTempView('observations')
        windows=spark.read.parquet(str(Path(prepared)/'windows.parquet'));windows.createOrReplaceTempView('windows')
        queries={
            'annual':"SELECT year, split, COUNT(*) AS observations, COUNT(DISTINCT event_id) AS events FROM observations GROUP BY year, split ORDER BY year",
            'windows':"SELECT split, context, COUNT(*) AS windows, COUNT(DISTINCT event_id) AS events FROM windows GROUP BY split, context ORDER BY split, context",
            'quality':"SELECT split, status, COUNT(*) AS observations, SUM(positive_pixels) AS positive_pixels, SUM(valid_pixels) AS valid_pixels, SUM(invalid_pixels) AS invalid_pixels, SUM(CASE WHEN positive_pixels=0 THEN 1 ELSE 0 END) AS no_detection_days FROM observations WHERE split IN ('train','val') GROUP BY split, status ORDER BY split, status",
            'duplicate_keys':"SELECT event_id,date,COUNT(*) AS n FROM observations GROUP BY event_id,date HAVING COUNT(*)>1",
            'annual_leakage':"SELECT COUNT(*) AS violations FROM observations WHERE (split='train' AND year NOT IN (2018,2019)) OR (split='val' AND year<>2020)",
            'window_join':"SELECT w.split, COUNT(*) AS matched_targets FROM windows w JOIN observations o ON w.event_id=o.event_id AND w.target_date=o.date WHERE w.context=3 GROUP BY w.split ORDER BY w.split"
        }
        results={};times={}
        for name,query in queries.items():
            ts=time.perf_counter();table=spark.sql(query);rows=[r.asDict() for r in table.collect()];times[name]=time.perf_counter()-ts;results[name]=rows
            if rows:pq.write_table(pa.Table.from_pylist(rows),out/f'{name}.parquet',compression='snappy')
        assert sum(r['events'] for r in results['annual'])==607 and sum(r['observations'] for r in results['annual'])==13607
        assert not results['duplicate_keys'] and results['annual_leakage'][0]['violations']==0
        report=dict(status='SPARK_EXECUTED',master=spark.sparkContext.master,spark_version=spark.version,distributed_cluster=False,jvm_start_seconds=init,query_seconds=times,processing_seconds=time.perf_counter()-t,total_seconds=time.perf_counter()-started,results=results,queries=queries,source_hashes={f:sha(Path(prepared)/f) for f in ('observations.parquet','windows.parquet')},scope='2021 metadata inventory only; quality from existing certified train/val counts; no new pixel audit',materialization='real Spark SQL collect -> small PyArrow Parquet tables on Windows')
        save(out/'report.json',report);save(out/'queries.json',queries);print(json.dumps(report),flush=True)
    finally:spark.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',default='reports/full_corpus_v1/prepared_spark_v1');p.add_argument('--out',required=True);a=p.parse_args();run(a.prepared,a.out)
