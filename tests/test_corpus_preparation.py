import io
import json
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from geofirex.corpus import (BudgetRangeFile, acquire_members, index_zip,
                            inventory_local, safe_entry, sha256_file)
from geofirex.event_cache import EventCache, build_event_cache
from geofirex.real_constants import RAW_BAND_NAMES
from geofirex.remote_archive import RangeResponseError
from geofirex.unet_data import Normalizer


def archive(tmp_path):
    path=tmp_path/"archive.zip"
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("2018/fire_1/2018-01-01.tif",b"real bytes for fixture 1")
        zf.writestr("2018/fire_1/2018-01-02.tif",b"real bytes for fixture 2")
    return path


def test_acquisition_resume_uses_crc_and_preserves_same_size_corruption(tmp_path):
    with zipfile.ZipFile(archive(tmp_path)) as zf:
        index=index_zip(zf)
        destination=tmp_path/"out"
        first=acquire_members(zf,index["entries"],destination,max_files=1)
        assert len(first["written"])==1
        second=acquire_members(zf,index["entries"],destination)
        assert len(second["written"])==len(second["verified_existing"])==1
        assert inventory_local(index,destination)["complete"]
        file=destination/index["entries"][0]["path"]
        file.write_bytes(b"X"*file.stat().st_size)
        digest=sha256_file(file)
        with pytest.raises(ValueError,match="preserved"):
            acquire_members(zf,index["entries"],destination)
        assert sha256_file(file)==digest


def test_acquisition_interruption_does_not_publish_partial_tiff(tmp_path):
    with zipfile.ZipFile(archive(tmp_path)) as zf:
        entries=index_zip(zf)["entries"]
        with patch.object(zf,"open",side_effect=OSError("interrupted")):
            with pytest.raises(OSError):
                acquire_members(zf,entries,tmp_path/"out")
    assert not list((tmp_path/"out").rglob("*.tif"))


@pytest.mark.parametrize("name",["../escape.tif","2018/fire_1/../../escape.tif","2018/fire_1/2018-02-30.tif"])
def test_unsafe_and_invalid_archive_entries_rejected(name):
    with pytest.raises(ValueError):
        safe_entry(name)


class Response:
    def __init__(self,status=206,body=b"abcd",content_range="bytes 0-3/100"):
        self.status=status; self.headers={"Content-Range":content_range}
        self.body=io.BytesIO(body); self.reads=[]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,n):
        self.reads.append(n)
        return self.body.read(n)


def test_budgeted_range_refuses_200_without_reading_body():
    response=Response(status=200,body=b"X"*10000)
    with patch("urllib.request.urlopen",return_value=response):
        with pytest.raises(RangeResponseError,match="206"):
            BudgetRangeFile(10,size=100,retries=1).read(4)
    assert response.reads==[]


def test_budget_stops_before_network_and_bounds_body_read():
    stream=BudgetRangeFile(3,size=100,retries=1)
    with patch("urllib.request.urlopen") as request:
        with pytest.raises(RuntimeError,match="budget"):
            stream.read(4)
        request.assert_not_called()
    response=Response(body=b"abcdefghijk")
    with patch("urllib.request.urlopen",return_value=response):
        with pytest.raises(RangeResponseError,match="length"):
            BudgetRangeFile(10,size=100,retries=1).read(4)
    assert response.reads==[5] and response.body.tell()==5


def test_budget_counts_retry_reservations():
    stream=BudgetRangeFile(7,size=100,retries=2)
    with patch("urllib.request.urlopen",side_effect=OSError("failed")) as request:
        with patch("geofirex.corpus.time.sleep"):
            with pytest.raises(RuntimeError,match="budget"):
                stream.read(4)
    assert request.call_count==1 and stream.bytes_reserved==5


def test_archive_mismatch_cannot_be_acquired(tmp_path):
    with zipfile.ZipFile(archive(tmp_path)) as zf:
        entries=index_zip(zf)["entries"]
        entries[0]["crc32"]="00000000"
        with pytest.raises(ValueError,match="differs"):
            acquire_members(zf,entries,tmp_path/"out")


def cache_fixture(tmp_path):
    rows=[]; arrays=[]
    for day in (1,2):
        path=tmp_path/f"2018-01-0{day}.tif"
        raw=np.ones((23,10,14),dtype=np.float32)
        raw[12,0,0]=np.nan
        raw[13]=90
        raw[14]=100
        raw[22]=np.nan; raw[22,3,3]=2036
        with rasterio.open(path,"w",driver="GTiff",count=23,width=14,height=10,dtype="float32",
                           crs="EPSG:32610",transform=from_origin(0,0,375,375)) as ds:
            ds.write(raw); ds.descriptions=RAW_BAND_NAMES
        rows.append({"event_id":"2018/fire_1","date":f"2018-01-0{day}","relative_path":str(path)})
        arrays.append(raw)
    return rows,arrays


def test_memmap_cache_is_unnormalized_readonly_and_matches_feature_encoding(tmp_path):
    rows,raw=cache_fixture(tmp_path)
    before=[sha256_file(r["relative_path"]) for r in rows]
    destination=tmp_path/"cache"
    metadata=build_event_cache(rows,tmp_path,destination,expected_dates=[r["date"] for r in rows])
    cache=EventCache(destination)
    try:
        assert isinstance(cache.numeric,np.memmap) and not cache.numeric.flags.writeable
        np.testing.assert_array_equal(cache.numeric[0],raw[0][[12,13,14]])
        norm=Normalizer([12,14],[2,100],[1,1],[1,1],[],include_landcover=True)
        a=norm.transform(raw[0][:,:8,:8])
        b=norm.transform(cache.raw_patch(0,0,0,8))
        np.testing.assert_array_equal(a,b)
        assert np.isnan(cache.raw_patch(0,0,0,8)[0]).all()
    finally:
        cache.close()
    assert [sha256_file(r["relative_path"]) for r in rows]==before
    assert metadata==build_event_cache(rows,tmp_path,destination,expected_dates=[r["date"] for r in rows])


def test_incomplete_event_cache_refused(tmp_path):
    rows,_=cache_fixture(tmp_path)
    with pytest.raises(ValueError,match="incomplete"):
        build_event_cache(rows[:1],tmp_path,tmp_path/"cache",expected_dates=[r["date"] for r in rows])


def test_parquet_metadata_roundtrip_has_qc_and_real_coordinates(tmp_path):
    pa=pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from scripts.export_observation_metadata import observation_metadata
    rows,_=cache_fixture(tmp_path)
    for row in rows:
        row["year"]="2018"
    records=observation_metadata(rows,tmp_path)
    path=tmp_path/"observations.parquet"
    pq.write_table(pa.Table.from_pylist(records),path)
    assert pq.read_table(path).to_pylist()==records
    assert records[0]["slope_finite_count"]==139
    assert records[0]["fire_pixels"]==1
    assert -180<=records[0]["raster_center_longitude"]<=180
