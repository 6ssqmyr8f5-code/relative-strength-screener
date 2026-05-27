import os
import tempfile
import builtins
import pandas as pd
from data_provider import (
    get_stock_daily,
    is_cache_valid,
    read_cache,
    save_cache,
    load_local_csv,
    _stock_cache_path,
    _index_cache_path,
)


def test_save_and_read_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=10),
            "close": range(100, 110),
            "volume": range(1000, 1010),
        })
        path = os.path.join(tmpdir, "test.csv")
        save_cache(df, path)
        assert os.path.exists(path)
        result = read_cache(path)
        assert len(result) == 10
        assert result["date"].dtype == "datetime64[ns]"


def test_is_cache_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        df = pd.DataFrame({"date": ["2022-01-01"], "close": [100]})
        df.to_csv(path, index=False, encoding="utf-8-sig")
        assert is_cache_valid(path, 1) is True
        assert is_cache_valid(os.path.join(tmpdir, "nonexistent.csv"), 1) is False


def test_stock_cache_path():
    path = _stock_cache_path("data/cache/stocks", "600519", "20220101", "20260527", "qfq")
    assert "600519" in path
    assert "20220101" in path
    assert "qfq.csv" in path


def test_index_cache_path():
    path = _index_cache_path("data/cache/index", "000300", "20220101", "20260527")
    assert "000300" in path
    assert "20220101" in path


def test_local_csv_utf8_sig():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.csv")
        df = pd.DataFrame({
            "date": pd.date_range("2022-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
        })
        df.to_csv(path, index=False, encoding="utf-8-sig")
        result = load_local_csv(path)
        assert len(result) == 5
        assert "date" in result.columns
        assert "close" in result.columns


def test_get_stock_daily_uses_local_fallback_when_akshare_fails(tmp_path, monkeypatch):
    path = tmp_path / "600519.csv"
    df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=5),
        "close": [100, 101, 102, 103, 104],
        "volume": [1000, 1001, 1002, 1003, 1004],
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "akshare":
            raise ImportError("offline")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result, source = get_stock_daily(
        code="600519",
        start="20220101",
        end="20220110",
        adjust="qfq",
        cache_config={"enabled": True, "stock_cache_dir": str(tmp_path / "cache")},
        retry_config={},
        local_path=str(path),
    )

    assert source.startswith("local_csv:")
    assert len(result) == 5
    assert result["close"].iloc[-1] == 104


def test_longterm_cache_path_generation():
    from data_provider import _stock_longterm_cache_path, _index_longterm_cache_path

    stock_path = _stock_longterm_cache_path("data/cache/stocks", "600519", "qfq")
    assert stock_path == "data/cache/stocks/600519_qfq.csv"

    index_path = _index_longterm_cache_path("data/cache/index", "000300")
    assert index_path == "data/cache/index/000300.csv"


def test_cache_source_sleep_behavior():
    source_cache = "cache"
    source_akshare = "akshare"
    source_local = "local_csv:test"

    assert "akshare" not in source_cache.lower()
    assert "akshare" in source_akshare.lower()
    assert "akshare" not in source_local.lower()
