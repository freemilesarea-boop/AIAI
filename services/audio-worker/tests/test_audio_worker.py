from luber_audio_worker.worker import QUEUE_NAME, WorkerSettings, ping


async def test_ping_task_returns_payload():
    result = await ping({"worker_id": "test-worker"}, payload="hello")
    assert result == "hello"


def test_worker_settings_shape():
    assert QUEUE_NAME == "luber:audio"
    assert ping in WorkerSettings.functions
