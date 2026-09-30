"""
The job API in front of the Modal GPU workers.

It speaks the protocol of RunPod serverless endpoints, so the server's ``JobEndpoint`` client
works with either host:

    POST /{worker}/run            {"input": {...}}  ->  {"id", "status": "IN_QUEUE"}
    POST /{worker}/runsync        same, but waits for up to a minute for the output
    GET  /{worker}/status/{id}    ->  {"id", "status", "output"?, "error"?}
    POST /{worker}/cancel/{id}    ->  {"id", "status": "CANCELLED"}

Every request needs ``Authorization: Bearer <ORAINGE_WORKER_TOKEN>``.
"""

from __future__ import annotations

import contextlib
import hmac
import re
import sys
import traceback
from typing import Any, Iterator, Mapping, Protocol

from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

MIN_TOKEN_LENGTH = 32
RUNSYNC_WAIT_S = 60.0  # Modal web requests are cut off after 150 s; poll after this
MAX_ERROR_LENGTH = 300


class JobNotFound(LookupError):
    """The job id is unknown, or its result has expired."""


class JobCancelled(Exception):
    """The job was cancelled before it finished."""


class JobFailed(Exception):
    """The job failed. The message is a one-line reason for the caller."""


class WorkersUnavailable(Exception):
    """Modal couldn't be reached. Says nothing about the job itself, so the caller should retry."""


class Calls(Protocol):
    """Starts jobs and reads their results. ``ModalCalls`` is the real one; tests use a fake."""

    @property
    def workers(self) -> tuple[str, ...]: ...

    def spawn(self, worker: str, job: dict) -> str:
        """Queue ``job`` on ``worker`` and return its id."""
        ...

    def result(self, job_id: str, timeout: float) -> Any:
        """
        The job's output. Raises ``TimeoutError`` while it is still running, and
        ``JobNotFound``, ``JobCancelled``, ``JobFailed`` or ``WorkersUnavailable``. Any other
        exception also means the job failed.
        """
        ...

    def cancel(self, job_id: str) -> None: ...


@contextlib.contextmanager
def _modal_outages() -> Iterator[None]:
    import modal.exception as mx

    try:
        yield
    except (
        mx.ClientClosed,
        mx.ConnectionError,
        mx.InternalError,
        mx.ResourceExhaustedError,
        mx.ServiceError,
    ) as err:
        raise WorkersUnavailable(f"{type(err).__name__}: {err}") from err


class ModalCalls:
    """``Calls`` backed by Modal: a job is a ``modal.FunctionCall`` and its id is the call's id."""

    CALL_ID = re.compile(r"fc-[A-Za-z0-9]{1,64}")

    def __init__(self, functions: Mapping[str, Any]) -> None:
        # Worker name -> Modal function (or class method) that takes the job dict
        self._functions = dict(functions)

    @property
    def workers(self) -> tuple[str, ...]:
        return tuple(self._functions)

    def spawn(self, worker: str, job: dict) -> str:
        with _modal_outages():
            return self._functions[worker].spawn(job).object_id

    def result(self, job_id: str, timeout: float) -> Any:
        import modal.exception as mx

        call = self._call(job_id)
        with _modal_outages():
            try:
                # Raises the builtin TimeoutError while the call is still running
                return call.get(timeout=timeout)
            except (mx.NotFoundError, mx.InvalidError, mx.OutputExpiredError) as err:
                raise JobNotFound(job_id) from err
            except mx.RemoteError as err:
                # Modal reports a cancelled call as a terminated result without any details
                if not str(err):
                    raise JobCancelled(job_id) from err
                raise
            except mx.ExecutionError as err:
                # This image has no torch or transformers, so their errors can't be unpickled
                # here and arrive as the remote traceback. Its last line names the error.
                lines = [line.strip() for line in str(err).splitlines() if line.strip()]
                raise JobFailed(lines[-1] if lines else "the job failed") from err

    def cancel(self, job_id: str) -> None:
        import modal.exception as mx

        call = self._call(job_id)
        with _modal_outages():
            try:
                call.cancel()
            except (mx.NotFoundError, mx.InvalidError) as err:
                raise JobNotFound(job_id) from err

    def _call(self, job_id: str) -> Any:
        import modal

        if not self.CALL_ID.fullmatch(job_id):
            raise JobNotFound(job_id)
        return modal.FunctionCall.from_id(job_id)


def _log(job_id: str, err: BaseException) -> None:
    print(f"[orainge] job {job_id}:", file=sys.stderr)
    traceback.print_exception(err, file=sys.stderr)


def _reason(err: Exception) -> str:
    """One line for the caller; the full error goes to the logs."""
    if isinstance(err, JobFailed):
        text = str(err)
    else:
        first = next((line.strip() for line in str(err).splitlines() if line.strip()), "")
        text = f"{type(err).__name__}: {first}" if first else type(err).__name__
    return text[:MAX_ERROR_LENGTH]


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail="Modal can't be reached right now; try again")


def job_state(calls: Calls, job_id: str, wait: float) -> dict:
    """The RunPod-style status object for a job, waiting up to ``wait`` seconds for it to finish."""
    try:
        output = calls.result(job_id, wait)
    except TimeoutError:
        return {"id": job_id, "status": "IN_PROGRESS"}
    except JobNotFound:
        raise HTTPException(status_code=404, detail="unknown job, or its result has expired") from None
    except JobCancelled:
        return {"id": job_id, "status": "CANCELLED"}
    except WorkersUnavailable as err:
        _log(job_id, err)
        raise _unavailable() from None
    except Exception as err:  # noqa: BLE001 - e.g. the container crashed or timed out
        _log(job_id, err)
        return {"id": job_id, "status": "FAILED", "error": _reason(err)}
    # Like RunPod, a worker that returns {"error": ...} failed the job
    if isinstance(output, dict) and output.get("error"):
        return {"id": job_id, "status": "FAILED", "error": str(output["error"])}
    return {"id": job_id, "status": "COMPLETED", "output": output}


class RequireToken:
    """
    Checks the bearer token before anything reads the request. As a FastAPI dependency it would
    run only after the body was parsed, so strangers could make the API parse large uploads.
    """

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http":
            supplied = next((value for name, value in scope["headers"] if name == b"authorization"), b"")
            if not hmac.compare_digest(supplied, self.expected):
                response = JSONResponse({"detail": "missing or wrong bearer token"}, status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def app_for_token(token: str | None, calls: Calls) -> FastAPI:
    """
    The API for the token in the environment. With a missing or short token it still starts, but
    answers every request with the reason: Modal keeps restarting a web container that fails to
    start, so callers would only see their requests hang.
    """
    token = (token or "").strip()  # a pasted secret often ends in a line break
    if len(token) >= MIN_TOKEN_LENGTH:
        return create_app(token, calls)
    reason = (
        "The job API is off: ORAINGE_WORKER_TOKEN in the Modal secret orainge-worker-token must be "
        f"at least {MIN_TOKEN_LENGTH} characters"
    )
    print(f"[orainge] {reason}", file=sys.stderr)
    app = FastAPI(title="Orainge AI workers", docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def switched_off(path: str) -> JSONResponse:
        return JSONResponse({"detail": reason}, status_code=503)

    return app


def create_app(token: str, calls: Calls, runsync_wait: float = RUNSYNC_WAIT_S) -> FastAPI:
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(f"ORAINGE_WORKER_TOKEN must be at least {MIN_TOKEN_LENGTH} characters")

    # No /docs or /openapi.json: the URL is public, so don't describe the API to strangers
    app = FastAPI(title="Orainge AI workers", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequireToken, token=token)

    def worker_name(worker: str) -> str:
        if worker not in calls.workers:
            raise HTTPException(status_code=404, detail=f"unknown worker {worker!r}")
        return worker

    def spawn(worker: str, body: Any) -> str:
        if not isinstance(body, dict) or not isinstance(body.get("input"), dict):
            raise HTTPException(status_code=400, detail='the body must be {"input": {...}}')
        try:
            return calls.spawn(worker, {"input": body["input"]})
        except WorkersUnavailable as err:
            _log("(new)", err)
            raise _unavailable() from None

    @app.get("/")
    def index() -> dict:
        return {"workers": list(calls.workers)}

    @app.post("/{worker}/run")
    def run(worker: str = Depends(worker_name), body: Any = Body(default=None)) -> dict:
        return {"id": spawn(worker, body), "status": "IN_QUEUE"}

    @app.post("/{worker}/runsync")
    def runsync(worker: str = Depends(worker_name), body: Any = Body(default=None)) -> dict:
        return job_state(calls, spawn(worker, body), runsync_wait)

    @app.get("/{worker}/status/{job_id}")
    def status(job_id: str, worker: str = Depends(worker_name)) -> dict:
        return job_state(calls, job_id, 0)

    @app.post("/{worker}/cancel/{job_id}")
    def cancel(job_id: str, worker: str = Depends(worker_name)) -> dict:
        try:
            calls.cancel(job_id)
        except JobNotFound:
            raise HTTPException(status_code=404, detail="unknown job") from None
        except WorkersUnavailable as err:
            _log(job_id, err)
            raise _unavailable() from None
        return {"id": job_id, "status": "CANCELLED"}

    return app
