import pandas as pd

from main import select_chart_items
from utils import should_throttle_after_source


def test_should_throttle_after_network_sources_only():
    assert should_throttle_after_source("cache") is False
    assert should_throttle_after_source("local_csv:data/stocks/600519.csv") is False
    assert should_throttle_after_source("akshare_force_refresh") is True
    assert should_throttle_after_source("cache_incremental_update") is True
    assert should_throttle_after_source("cache_stale") is True
    assert should_throttle_after_source("", had_error=True) is True


def test_select_chart_items_follows_sorted_candidates_and_limit():
    candidates = pd.DataFrame(
        {
            "code": ["000002", "000001", "000003"],
            "score": [90, 80, 70],
        }
    )
    chart_data = {
        "000001": ("000001", "one", {"score": 80}, pd.DataFrame()),
        "000002": ("000002", "two", {"score": 90}, pd.DataFrame()),
        "000004": ("000004", "four", {"score": 60}, pd.DataFrame()),
    }

    selected = select_chart_items(candidates, chart_data, max_images=2)

    assert [item[0] for item in selected] == ["000002", "000001"]
