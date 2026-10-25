from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.models.schemas import GraphPayload, IngestResponse, QueryRequest, QueryResponse

router = APIRouter()

ALLOWED = {".pdf", ".md", ".txt"}
MAX_UPLOAD = 50 * 1024 * 1024


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, file: UploadFile) -> IngestResponse:
    suffix = "." + (file.filename or "").rsplit(".", 1)[-1].lower()
    if suffix not in ALLOWED:
        raise HTTPException(415, f"unsupported type {suffix}")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large")
    result = await request.app.state.pipeline.run(file.filename, data)
    return IngestResponse(**result)


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    if not body.question.strip():
        raise HTTPException(422, "empty question")
    return await request.app.state.query_service.answer(body.question)


@router.get("/graph", response_model=GraphPayload)
async def graph(request: Request, limit: int = 300) -> GraphPayload:
    return await request.app.state.graph.overview(min(limit, 1000))


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
