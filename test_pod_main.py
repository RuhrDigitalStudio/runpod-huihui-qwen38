import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import pod_main


class ContextSelectionTest(unittest.TestCase):
    def test_context_scales_with_aggregate_vram(self):
        self.assertEqual(pod_main.context_length_for_vram(24), 32768)
        self.assertEqual(pod_main.context_length_for_vram(32), 65536)
        self.assertEqual(pod_main.context_length_for_vram(40), 131072)
        self.assertEqual(pod_main.context_length_for_vram(48), 262144)
        self.assertEqual(pod_main.context_length_for_vram(96), 262144)

    @patch("pod_main.urlopen")
    def test_warmup_loads_model_with_persistent_keep_alive(self, urlopen):
        response = MagicMock()
        response.status = 200
        urlopen.return_value.__enter__.return_value = response

        self.assertTrue(pod_main.warm_model("huihui-test"))

        request = urlopen.call_args.args[0]
        payload = __import__("json").loads(request.data)
        self.assertEqual(payload["model"], "huihui-test")
        self.assertEqual(payload["keep_alive"], -1)
        self.assertEqual(payload["options"]["num_predict"], 1)


class StreamingProxyTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        os.environ["POD_API_SECRET"] = "test-secret"
        pod_main.model_ready.set()
        self.original_base_url = pod_main.OLLAMA_BASE_URL
        self.original_interval = pod_main.STREAM_HEARTBEAT_INTERVAL

    async def asyncTearDown(self):
        pod_main.OLLAMA_BASE_URL = self.original_base_url
        pod_main.STREAM_HEARTBEAT_INTERVAL = self.original_interval

    async def test_stream_sends_heartbeats_while_upstream_is_thinking(self):
        async def delayed_response(request):
            await request.read()
            await asyncio.sleep(0.12)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b"event: response.completed\ndata: {}\n\n")
            return response

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/responses", delayed_response)
        upstream = TestServer(upstream_app)
        await upstream.start_server()

        pod_main.OLLAMA_BASE_URL = str(upstream.make_url("/")).rstrip("/")
        pod_main.STREAM_HEARTBEAT_INTERVAL = 0.03

        proxy_app = web.Application()
        proxy_app.router.add_route("*", "/{path:.*}", pod_main.proxy)
        client = TestClient(TestServer(proxy_app))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/responses",
                json={"model": "test", "input": "review", "stream": True},
            )
            content = await response.read()
        finally:
            await client.close()
            await upstream.close()

        self.assertEqual(response.status, 200)
        self.assertTrue(content.startswith(b": connected\n\n"))
        self.assertIn(b": keep-alive\n\n", content)
        self.assertIn(b"event: response.completed", content)


if __name__ == "__main__":
    unittest.main()
