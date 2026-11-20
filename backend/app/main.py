from __future__ import annotations
import uuid
from fastapi import FastAPI,Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.container import Container
from app.core.settings import Settings

def create_app(data_dir:str|None=None):
    settings=Settings.load()
    app=FastAPI(title="TreeMindDB",version="0.1.0")
    app.state.container=Container(data_dir,settings)
    app.add_middleware(CORSMiddleware,allow_origins=settings.cors_origins,allow_methods=["*"],allow_headers=["*"])
    @app.middleware("http")
    async def correlation_id(request:Request,call_next):
        response=await call_next(request); response.headers["X-Request-ID"]=request.headers.get("X-Request-ID",str(uuid.uuid4())); return response
    app.include_router(router); return app
app=create_app()
