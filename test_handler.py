from handler import _normalize_job_input


def test_openai_post():
    assert _normalize_job_input(
        {"openai_route": "/v1/responses", "openai_input": {"input": "hi"}}
    ) == ("/v1/responses", "POST", {"input": "hi"})


def test_models_get():
    assert _normalize_job_input({"openai_route": "/v1/models"}) == (
        "/v1/models",
        "GET",
        None,
    )
