"""자신의 심화 구현을 검사합니다. 실제 모델 품질 테스트가 아닙니다."""
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]))
from coursekit.io import DATA, read_json
from retry_policy import run_with_retry, repair_prompt
from coursekit.retry import run_fixture_scenarios


def test_call_bounds():
    results=run_fixture_scenarios(read_json(DATA/"validation_fixtures.json"),runner=run_with_retry)
    assert [r["calls"] for r in results]==[2,2,1,1]

def test_unknown_error_not_forwarded():
    with pytest.raises(ValueError):repair_prompt("original","PRIVATE_PATH_TEST")

def test_safe_summary():
    for error in ["json_error","schema_error"]:
        result=repair_prompt("same prompt",error)
        assert result.startswith("same prompt") and "PRIVATE_PATH_TEST" not in result
