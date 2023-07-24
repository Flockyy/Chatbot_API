from fastapi import FastAPI
import uvicorn
from src.core.config import settings
from src.api.api_v1.api import api_router
from fastapi.middleware.cors import CORSMiddleware

import sentry_sdk

sentry_sdk.init(
    dsn="https://555ce8b32f154b1fbc05484891fdc174@o4505475663200256.ingest.sentry.io/4505475664838656",
    max_breadcrumbs=50,
    debug=True,
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production,
    traces_sample_rate=1.0,
    send_default_pii=True,
)

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(api_router, prefix=settings.API_V1_STR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
