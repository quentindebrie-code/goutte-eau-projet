from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.weather import FEATURES, forecast, prepare_observations, risk_band, train


@pytest.fixture
def raw():
    dates = pd.date_range('2026-05-01', periods=151)
    return pd.DataFrame({'time': dates.strftime('%Y-%m-%d'), 'temperature_2m_max': 20.,
        'temperature_2m_min': 10., 'relative_humidity_2m_mean': 75.,
        'surface_pressure_mean': 1010., 'wind_speed_10m_max': 15.,
        'cloud_cover_mean': 50., 'precipitation_sum': np.arange(len(dates)) % 2 * 3.})


@pytest.fixture
def model():
    m = MagicMock()
    m.classes_ = np.array([0,1])
    m.predict_proba.return_value = np.array([[.53,.47]])
    return m


def test_unknown_precipitation_is_not_dry(raw):
    raw.loc[1, 'precipitation_sum'] = np.nan
    df,_ = prepare_observations(raw)
    assert pd.isna(df.loc[0,'rain_tomorrow'])
    assert pd.isna(df.iloc[-1]['rain_tomorrow'])
    assert len(df) == len(raw)


def test_rain_threshold_is_strict(raw):
    raw.loc[1,'precipitation_sum'] = .5
    raw.loc[2,'precipitation_sum'] = .50001
    df,_ = prepare_observations(raw)
    assert df.loc[0,'rain_tomorrow'] == 0
    assert df.loc[1,'rain_tomorrow'] == 1


def test_gap_does_not_shift_outcome(raw):
    df,_ = prepare_observations(raw.drop(index=1))
    assert pd.isna(df.loc[0,'rain_tomorrow'])


@pytest.mark.parametrize('column,value', [('surface_pressure_mean',None),
    ('surface_pressure_mean',2000),('temperature_2m_max',120),
    ('relative_humidity_2m_mean',150)])
def test_invalid_features_suspend_forecast(raw,model,column,value):
    raw.loc[0,column] = value
    df,q = prepare_observations(raw)
    assert not q.loc[0,'valide']
    assert forecast(model,df,date(2026,5,2),q) is None
    model.predict_proba.assert_not_called()


def test_duplicate_dates_are_excluded(raw):
    raw = pd.concat([raw,raw.iloc[[0]]],ignore_index=True)
    df,q = prepare_observations(raw)
    assert len(q.loc[~q['valide']]) == 2
    assert '2026-05-01' not in set(df['date'])


def test_target_joins_previous_day_features(raw,model):
    raw.loc[0,'temperature_2m_max'] = 22
    df,q = prepare_observations(raw)
    r = forecast(model,df,date(2026,5,2),q)
    assert r['date_features'] == '2026-05-01'
    assert r['date_cible'] == '2026-05-02'
    assert r['source_features'] == 'observee'
    assert model.predict_proba.call_args.args[0][0,0] == 22


def test_unknown_historical_date_is_unavailable(raw,model):
    df,q = prepare_observations(raw.drop(index=1))
    assert forecast(model,df,date(2026,5,3),q) is None


def test_empty_observations_suspend_forecast(raw,model):
    df,q = prepare_observations(raw.iloc[:0])
    assert forecast(model,df,date(2026,5,3),q) is None
    model.predict_proba.assert_not_called()


def test_out_of_calendar_observation_is_rejected(raw):
    raw.loc[0,'time'] = '2027-05-01'
    df,q = prepare_observations(raw, as_of=date(2026,10,1))
    assert not q.loc[0,'valide']
    assert '2027-05-01' not in set(df['date'])


def test_sqlite_pipeline_never_fills_unknown_targets(raw):
    from src.collect import clean_and_transform
    raw.loc[1,'precipitation_sum'] = None
    df = clean_and_transform(raw.drop(index=3))
    assert '2026-05-01' not in set(df['date'])
    assert '2026-05-03' not in set(df['date'])
    assert raw.iloc[-1]['time'] not in set(df['date'])


def test_future_proxy_is_explicit(raw,model):
    df,q = prepare_observations(raw)
    r = forecast(model,df,date(2027,5,2),q)
    assert r['source_features'] == 'proxy_saisonnier'
    assert r['confidence'] == 'non calibrée'


@pytest.mark.parametrize('p,band',[(0,'faible'),(.349999,'faible'),(.35,'modere'),(.59999,'modere'),(.6,'eleve'),(1,'eleve')])
def test_risk_boundaries(p,band):
    assert risk_band(p) == band


@pytest.mark.parametrize('p',[None,np.nan,-.1,1.1])
def test_invalid_probability_is_rejected(p):
    with pytest.raises(ValueError): risk_band(p)


def test_training_uses_only_known_targets_and_changes_version(raw):
    df,_ = prepare_observations(raw)
    _,m1 = train(df)
    assert m1['n_train'] + m1['n_test'] == 150
    assert np.array(m1['confusion_matrix']).sum() == m1['n_test']
    changed = df.copy()
    changed.loc[0,'temp_max'] = 21
    _,m2 = train(changed)
    assert m1['version_modele'] != m2['version_modele']


def test_streamlit_smoke_and_updated_data_cache(raw):
    from streamlit.testing.v1 import AppTest
    response = MagicMock()
    response.json.return_value = {'daily':raw.to_dict(orient='list')}
    with patch('requests.get',return_value=response):
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'src/app.py', default_timeout=30).run()
        assert not app.exception
        assert any('Modèle prêt' in v.value for v in app.success)
        version_before = next(v.value for v in app.caption if 'modèle rf-' in v.value)
        app.button[1].click().run()
        assert not app.exception
        assert any('source : ' in v.value for v in app.caption)
        changed = raw.copy()
        changed.loc[0,'temperature_2m_max'] = 21.
        response.json.return_value = {'daily':changed.to_dict(orient='list')}
        app.button[0].click().run()
        assert not app.exception
        version_after = next(v.value for v in app.caption if 'modèle rf-' in v.value)
        assert version_before != version_after
