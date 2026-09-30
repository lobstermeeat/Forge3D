"""The RunPod-compatible job API in front of the Modal workers, with fake Modal calls."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from job_api import JobCancelled, JobFailed, JobNotFound, ModalCalls, WorkersUnavailable, create_app

TOKEN = "k" * 43
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeCalls:
    workers = ("trellis2", "reference")

    def __init__(self):
        self.spawned = []
        self.outcomes = {}  # job id -> output, or an exception to raise; missing means still running
        self.waits = []
        self.cancelled = []

    def spawn(self, worker, job):
        if job["input"].get("prompt") == "modal is down":
            raise WorkersUnavailable("ServiceError: unavailable")
        self.spawned.append((worker, job))
        return f"fc-{len(self.spawned)}"

    def result(self, job_id, timeout):
        self.waits.append((job_id, timeout))
        outcome = self.outcomes.get(job_id, TimeoutError())
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def cancel(self, job_id):
        if job_id == "fc-gone":
            raise JobNotFound(job_id)
        self.cancelled.append(job_id)


@pytest.fixture
def calls():
    return FakeCalls()


@pytest.fixture
def client(calls):
    return TestClient(create_app(TOKEN, calls, runsync_wait=5))


def test_every_route_needs_the_token(client, calls):
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": f"Bearer {TOKEN}x"}).status_code == 401
    assert client.get("/", headers={"Authorization": TOKEN}).status_code == 401
    assert client.post("/trellis2/run", json={"input": {}}).status_code == 401
    assert client.get("/trellis2/status/fc-1").status_code == 401
    assert client.post("/trellis2/cancel/fc-1").status_code == 401
    assert calls.spawned == [] and calls.cancelled == []
    assert client.get("/", headers=AUTH).json() == {"workers": ["trellis2", "reference"]}


def test_checks_the_token_before_reading_the_body(client, calls):
    # Strangers get a 401 even for bodies FastAPI couldn't parse
    response = client.post("/trellis2/run", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 401
    assert client.post("/nope/run", content=b"x" * 1024).status_code == 401
    # With the token, the same body is a client error
    response = client.post("/trellis2/run", content=b"{not json", headers={**AUTH, "content-type": "application/json"})
    assert response.status_code == 422

    # And the body is never read: a 50 MB upload is turned away unread
    messages = []

    async def receive():
        raise AssertionError("the body was read before the token was checked")

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/trellis2/run",
        "raw_path": b"/trellis2/run",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"content-length", b"52428800")],
        "client": ("203.0.113.9", 50000),
        "server": ("example.modal.run", 443),
    }
    asyncio.run(create_app(TOKEN, calls)(scope, receive, send))
    assert messages[0]["type"] == "http.response.start" and messages[0]["status"] == 401
    assert calls.spawned == []


def test_refuses_to_start_with_a_weak_token(calls):
    with pytest.raises(ValueError, match="at least 32 characters"):
        create_app("hunter2", calls)
    with pytest.raises(ValueError):
        create_app("", calls)


def test_run_queues_only_the_input(client, calls):
    body = {"input": {"image_url": "https://assets.example.com/cat.png", "mode": "preview"}, "webhook": "x"}
    response = client.post("/trellis2/run", json=body, headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"id": "fc-1", "status": "IN_QUEUE"}
    assert calls.spawned == [("trellis2", {"input": body["input"]})]


@pytest.mark.parametrize("body", [None, [], {"input": "cat.png"}, {"prompt": "a teapot"}])
def test_rejects_bodies_without_an_input_object(client, calls, body):
    response = client.post("/reference/run", json=body, headers=AUTH)
    assert response.status_code == 400
    assert calls.spawned == []


def test_unknown_workers_are_not_found(client, calls):
    assert client.post("/hunyuan/run", json={"input": {}}, headers=AUTH).status_code == 404
    assert client.get("/hunyuan/status/fc-1", headers=AUTH).status_code == 404
    assert calls.spawned == []


def test_status_reports_each_outcome_like_runpod(client, calls):
    output = {"request_id": "gen_1", "glb": {"key": "ai/gen_1/final-7.glb", "url": None}}
    calls.outcomes = {
        "fc-done": output,
        "fc-bad-input": {"error": "invalid input: mode must be one of ['final', 'preview']"},
        "fc-crashed": RuntimeError("container exited with code 137\n  File /root/modal_app.py, line 9"),
        "fc-oom": JobFailed("torch.OutOfMemoryError: CUDA out of memory"),
        "fc-cancelled": JobCancelled("fc-cancelled"),
        "fc-expired": JobNotFound("fc-expired"),
        "fc-outage": WorkersUnavailable("ServiceError: unavailable"),
    }

    def status(job_id):
        response = client.get(f"/trellis2/status/{job_id}", headers=AUTH)
        return response.status_code, response.json()

    assert status("fc-running") == (200, {"id": "fc-running", "status": "IN_PROGRESS"})
    assert status("fc-done") == (200, {"id": "fc-done", "status": "COMPLETED", "output": output})
    assert status("fc-bad-input") == (
        200,
        {
            "id": "fc-bad-input",
            "status": "FAILED",
            "error": "invalid input: mode must be one of ['final', 'preview']",
        },
    )
    # One line for the caller; the traceback stays in the logs
    assert status("fc-crashed") == (
        200,
        {"id": "fc-crashed", "status": "FAILED", "error": "RuntimeError: container exited with code 137"},
    )
    assert status("fc-oom") == (
        200,
        {"id": "fc-oom", "status": "FAILED", "error": "torch.OutOfMemoryError: CUDA out of memory"},
    )
    assert status("fc-cancelled") == (200, {"id": "fc-cancelled", "status": "CANCELLED"})
    assert status("fc-expired")[0] == 404
    # Modal being unreachable isn't the job's fault: the client can try again
    assert status("fc-outage")[0] == 503
    # Status never blocks
    assert {timeout for _, timeout in calls.waits} == {0}


def test_runsync_waits_for_the_output(client, calls):
    calls.outcomes["fc-1"] = {"images": [], "prompt": "a teapot", "seconds": 1.5}
    response = client.post("/reference/runsync", json={"input": {"prompt": "a teapot"}}, headers=AUTH)
    assert response.json() == {
        "id": "fc-1",
        "status": "COMPLETED",
        "output": {"images": [], "prompt": "a teapot", "seconds": 1.5},
    }
    assert calls.waits == [("fc-1", 5)]

    # Still running after the wait (e.g. a cold start): the client polls /status
    response = client.post("/reference/runsync", json={"input": {"prompt": "a lamp"}}, headers=AUTH)
    assert response.json() == {"id": "fc-2", "status": "IN_PROGRESS"}


def test_modal_outages_are_503_not_failures(client, calls):
    response = client.post("/reference/run", json={"input": {"prompt": "modal is down"}}, headers=AUTH)
    assert response.status_code == 503


def test_cancel(client, calls):
    response = client.post("/reference/cancel/fc-3", headers=AUTH)
    assert response.json() == {"id": "fc-3", "status": "CANCELLED"}
    assert calls.cancelled == ["fc-3"]
    assert client.post("/reference/cancel/fc-gone", headers=AUTH).status_code == 404


# ModalCalls: the translation from Modal's function calls and errors


class FakeFunctionCall:
    def __init__(self, outcome):
        self.outcome = outcome
        self.cancelled = False
        self.timeouts = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    def cancel(self, terminate_containers=False):
        self.cancelled = True


class FakeFunction:
    def __init__(self):
        self.jobs = []

    def spawn(self, job):
        self.jobs.append(job)
        return type("Call", (), {"object_id": "fc-01K6ABC"})()


@pytest.fixture
def modal_calls(monkeypatch):
    modal = pytest.importorskip("modal")
    registry = {}

    def from_id(job_id):
        if job_id not in registry:
            raise AssertionError(f"from_id({job_id!r}) was not expected")
        return registry[job_id]

    monkeypatch.setattr(modal.FunctionCall, "from_id", staticmethod(from_id))
    function = FakeFunction()
    return ModalCalls({"trellis2": function}), registry, function


def test_modal_calls_spawn_returns_the_call_id(modal_calls):
    calls, _, function = modal_calls
    assert calls.workers == ("trellis2",)
    assert calls.spawn("trellis2", {"input": {"mode": "preview"}}) == "fc-01K6ABC"
    assert function.jobs == [{"input": {"mode": "preview"}}]


def test_modal_calls_translate_modal_errors(modal_calls):
    from modal.exception import (
        ExecutionError,
        FunctionTimeoutError,
        NotFoundError,
        OutputExpiredError,
        RemoteError,
        ServiceError,
    )

    calls, registry, _ = modal_calls
    # What Modal raises for an error this image can't unpickle (see _process_result)
    unpicklable = ExecutionError(
        "Could not deserialize remote exception due to local error:\n"
        "No module named 'torch'\n"
        "This can happen if your local environment does not have the remote exception definitions.\n"
        "Here is the remote traceback:\n"
        'Traceback (most recent call last):\n  File "/root/modal_app.py", line 170, in load\n'
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
    )
    registry.update(
        {
            "fc-done": FakeFunctionCall({"seed": 7}),
            "fc-running": FakeFunctionCall(TimeoutError()),
            "fc-expired": FakeFunctionCall(OutputExpiredError()),
            "fc-unknown": FakeFunctionCall(NotFoundError("no such call")),
            "fc-terminated": FakeFunctionCall(RemoteError("")),
            "fc-crashed": FakeFunctionCall(RemoteError("Runner crashed")),
            "fc-slow": FakeFunctionCall(FunctionTimeoutError("Function timed out")),
            "fc-oom": FakeFunctionCall(unpicklable),
            "fc-outage": FakeFunctionCall(ServiceError("unavailable")),
        }
    )
    assert calls.result("fc-done", 0) == {"seed": 7}
    with pytest.raises(TimeoutError):
        calls.result("fc-running", 60)
    # Status polls must not block; runsync waits as long as it was told
    assert registry["fc-done"].timeouts == [0]
    assert registry["fc-running"].timeouts == [60]
    for job_id in ("fc-expired", "fc-unknown"):
        with pytest.raises(JobNotFound):
            calls.result(job_id, 0)
    with pytest.raises(JobCancelled):
        calls.result("fc-terminated", 0)
    with pytest.raises(JobFailed, match=r"^torch\.OutOfMemoryError: CUDA out of memory\. Tried"):
        calls.result("fc-oom", 0)
    with pytest.raises(WorkersUnavailable):
        calls.result("fc-outage", 0)
    # Anything else is a failure the API reports as FAILED
    with pytest.raises(RemoteError, match="Runner crashed"):
        calls.result("fc-crashed", 0)
    with pytest.raises(FunctionTimeoutError):
        calls.result("fc-slow", 0)


def test_modal_calls_reject_malformed_ids_without_asking_modal(modal_calls):
    calls, _, _ = modal_calls
    for job_id in ("", "../trellis2", "fc-", "sb-01K6ABC", "fc-01K6ABC/../x", "fc-01K6ABC\n"):
        with pytest.raises(JobNotFound):
            calls.result(job_id, 0)
        with pytest.raises(JobNotFound):
            calls.cancel(job_id)


def test_modal_calls_cancel(modal_calls):
    from modal.exception import NotFoundError

    calls, registry, _ = modal_calls
    registry["fc-running"] = FakeFunctionCall(TimeoutError())
    calls.cancel("fc-running")
    assert registry["fc-running"].cancelled

    class Missing(FakeFunctionCall):
        def cancel(self, terminate_containers=False):
            raise NotFoundError("no such call")

    registry["fc-missing"] = Missing(None)
    with pytest.raises(JobNotFound):
        calls.cancel("fc-missing")
