import pytest
import time
from fastapi.testclient import TestClient
from pathlib import Path
from app.api.live_models import AvailableModels
from app.api.main import app
from app.api.routers.segment import SegmentRequest

@pytest.fixture(scope="session", autouse=True)
def setup_app():
    with TestClient(app) as c:
        yield c 



SESSIONS_DIR = Path("sessions")
N_IMAGES = 5

def get_sessions(n: int):
    sessions = [p for p in SESSIONS_DIR.iterdir() if p.is_dir()]
    if len(sessions) < n:
        pytest.exit(f"Not enough session dirs. Needed {n}, found {len(sessions)}")
    return sessions[:n]

# single request to $server/segment/ 
def run_segmentation(session, client) -> tuple[dict, float]:
    req = SegmentRequest(
        session_id=session.name,
        model=AvailableModels.yolosam,
        regions=[],
        blackout=False,
        inverse_blackout=False,
        colorize=False,
    )
    start = time.perf_counter()
    response = client.post("/segment/", json=req.model_dump())
    latency = time.perf_counter() - start

    assert response.status_code == 200, f"Got {response.status_code}: {response.text}"
    data = response.json()
    assert "error" not in data, f"Error in response: {data}"
    return data, latency

# test latency per image. 
def test_segment_per_image_latency(setup_app):
    client= setup_app

    sessions = get_sessions(N_IMAGES)
    latencies = []
    

    for i, session in enumerate(sessions):
        data, latency = run_segmentation(session, client)
        latencies.append(latency)
        print(f"\n[{i+1}/{N_IMAGES}] session: {session.name[:8]}"
              f" | detections: {data.get('metadata', {}).get('detections', '?')}"
              f" | latency: {latency:.3f}s")

    avg = sum(latencies) / len(latencies)
    mn  = min(latencies)
    mx  = max(latencies)

    print(f"\n── Summary ({N_IMAGES} images) ──────────────")
    print(f"  avg:   {avg:.3f}s")
    print(f"  min:   {mn:.3f}s")
    print(f"  max:   {mx:.3f}s")
    print(f"  total: {sum(latencies):.3f}s")



# We load the model whenveer the server starts,
# therefore, we expect all requests after the 
# first to be significantly faster. Test that
def test_first_vs_subsequent_latency(setup_app):
    client= setup_app
    sessions = get_sessions(N_IMAGES)
    latencies = []

    for session in sessions:
        _, latency = run_segmentation(session, client)
        latencies.append(latency)

    first    = latencies[0]
    rest_avg = sum(latencies[1:]) / len(latencies[1:]) if len(latencies) > 1 else first
    ratio    = first / rest_avg if rest_avg > 0 else 1.0

    print(f"\n── First vs Subsequent ──────────────")
    print(f"  first request:      {first:.3f}s")
    print(f"  avg (subsequent):   {rest_avg:.3f}s")
    print(f"  ratio:  {ratio:.1f}x")


def test_segment_benchmark(benchmark):
    sessions = get_sessions(N_IMAGES)
    client= setup_app

    def benchmark_run():
        results = []
        for session in sessions:
            result, _ = run_segmentation(session, client)
            results.append(result)
        return results

    benchmark(benchmark_run)

