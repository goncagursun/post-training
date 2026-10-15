"""HTTP interface: authenticate, validate, persist, and return immediately."""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader

from post_training.auth import authenticate_key
from post_training.control_plane import JobResponse, TrainingRequest, create_queued_job, get_owned_job, request_cancellation
from post_training.database import make_engine, make_session_factory


def create_app(session_factory=None):
    @asynccontextmanager
    async def lifespan(app):
        engine = None
        if session_factory is None:
            engine = make_engine()
            app.state.sessions = make_session_factory(engine)
        else:
            app.state.sessions = session_factory
        try:
            yield
        finally:
            if engine is not None:
                engine.dispose()

    app = FastAPI(title="Post-training MVP", lifespan=lifespan)
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def database_session():
        with app.state.sessions() as session:
            yield session

    def authenticated_tenant(raw_key=Depends(key_header), session=Depends(database_session)):
        tenant_id = authenticate_key(session, raw_key)
        if tenant_id is None:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return tenant_id

    @app.post("/training/jobs", response_model=JobResponse, status_code=202)
    def submit_job(request: TrainingRequest, tenant_id=Depends(authenticated_tenant),
                   session=Depends(database_session)):
        job = create_queued_job(session, tenant_id, request)
        if job is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return job

    @app.get("/training/jobs/{job_id}", response_model=JobResponse)
    def job_status(job_id: str, tenant_id=Depends(authenticated_tenant),
                   session=Depends(database_session)):
        job = get_owned_job(session, tenant_id, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @app.post("/training/jobs/{job_id}/cancel", response_model=JobResponse, status_code=202)
    def cancel_job(job_id: str, tenant_id=Depends(authenticated_tenant),
                   session=Depends(database_session)):
        job = request_cancellation(session, tenant_id, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    return app
