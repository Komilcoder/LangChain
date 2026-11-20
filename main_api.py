import uvicorn
from fastapi import FastAPI, APIRouter, Depends
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from rag import load_documents, embed, answer, load_index

app = FastAPI()

documents = load_index()
# print(answer("How much did Malika Yusupova's trip cost?", documents))


class Question(BaseModel):
    question: str


@app.post("/api/ask/")
def ask(body:Question):
    return answer(body.question, documents)


@app.get("/api/v1/swagger/", include_in_schema=False)
async def swagger_ui():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="XS Auth",
    )

router = APIRouter(prefix="/agent")

# all_routers = [
#     agent_router,
#
# ]


# for router in all_routers:
#     app.include_router(router, prefix="/api/v1/auth")


if __name__ == "__main__":
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
