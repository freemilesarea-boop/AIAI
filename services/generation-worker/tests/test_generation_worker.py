from luber_generation_worker.worker import QUEUE_NAME, WorkerSettings, ping


async def test_ping_task_returns_payload():
    result = await ping({"worker_id": "test-worker"}, payload="hello")
    assert result == "hello"


def test_worker_settings_shape():
    assert QUEUE_NAME == "luber:generation"
    assert ping in WorkerSettings.functions
    assert WorkerSettings.max_jobs == 1
