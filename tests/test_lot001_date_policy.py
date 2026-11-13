"""Targeted calendar/split regression tests; no raster, network or ML imports."""
import csv
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from certify_kaggle_lot001_runtime import LOT001_2017_CONTEXT, validate_observation_path, validate_lot_dates


class Lot001DatePolicyTests(unittest.TestCase):
    def test_train_folder_and_observation_year_need_not_match(self):
        split = {'2018/fire_example':'train', '2019/fire_example':'train'}
        for path in ('2018/fire_example/2019-01-01.tif', '2019/fire_example/2018-12-31.tif'):
            self.assertIn(validate_observation_path(path, split), (2018,2019))

    def test_2020_and_2021_always_stop_even_with_context_approval(self):
        for year in (2020,2021):
            for approval in (False,True):
                path = f'2018/fire_21458836/{year}-01-01.tif'
                with self.assertRaisesRegex(ValueError, f'Forbidden observation year {year}'):
                    validate_observation_path(path, {'2018/fire_21458836':'train'}, allow_2017_context=approval)

    def test_2017_exception_is_exact_and_requires_approval(self):
        split = {'2018/fire_21458836':'train', '2018/fire_other':'train'}
        for path in LOT001_2017_CONTEXT:
            with self.assertRaisesRegex(ValueError, 'outside approved train dates'):
                validate_observation_path(path, split)
            self.assertEqual(validate_observation_path(path, split, allow_2017_context=True), 2017)
        for path in ('2018/fire_21458836/2017-12-28.tif','2018/fire_other/2017-12-29.tif'):
            with self.assertRaises(ValueError):
                validate_observation_path(path, split, allow_2017_context=True)

    def test_nontrain_event_and_unsafe_paths_still_stop(self):
        path = '2018/fire_21458836/2017-12-29.tif'
        for role in ('val','test',None):
            with self.assertRaisesRegex(ValueError, 'not assigned to train'):
                validate_observation_path(path, {'2018/fire_21458836':role}, allow_2017_context=True)
        for path in ('../2018/fire_x/2018-01-01.tif','D:/rasters/2018-01-01.tif',
                     '2021/fire_x/2018-01-01.tif','2018/fire_x/2018-99-99.tif'):
            with self.assertRaises(ValueError):
                validate_observation_path(path, {'2018/fire_x':'train'}, allow_2017_context=True)

    def test_actual_lot_has_only_three_pinned_2017_contexts(self):
        base = Path(__file__).resolve().parents[1]/'reports/recovery_20261107_v3/finalized/deployment/bundle/metadata'
        lot = json.loads((base/'lot-001.json').read_text())
        with (base/'fold_00.csv').open() as stream:
            split = {r['event_id']:r['split'] for r in csv.DictReader(stream)}
        with self.assertRaisesRegex(ValueError, 'Observation year 2017'):
            validate_lot_dates(lot, split)
        report = validate_lot_dates(lot, split, allow_2017_context=True)
        self.assertEqual(report['observation_year_counts'], {'2017':3, '2018':1266})
        self.assertEqual({r['path'] for r in report['folder_date_year_mismatches']}, LOT001_2017_CONTEXT)
        self.assertEqual((len(lot['files']),len(lot['events'])), (1269,57))
        self.assertTrue(all(split[e]=='train' for e in lot['events']))


if __name__ == '__main__':
    unittest.main()
