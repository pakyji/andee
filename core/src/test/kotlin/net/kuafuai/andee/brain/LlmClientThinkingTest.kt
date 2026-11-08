package net.kuafuai.andee.brain

import com.sun.net.httpserver.HttpServer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.net.InetSocketAddress

/**
 * 2026-11-08: `complete()` sent DeepSeek's `thinking` / `reasoning_effort`
 * on every request, even with Deep thinking off — and OpenAI, Gemini and Meta
 * all 400 on an unknown `thinking` parameter, including
 * `{"type":"disabled"}`. The pair is now sent only to a DeepSeek-shaped
 * backend (DeepSeek host, or a DeepSeek model name on any host — which is
 * what keeps CodeFlying working, since its gateway serves `deepseek-flash`).
 *
 * Every case below talks to the local test server: the base URL's path
 * segment stands in for the provider's host (the gate is a substring check
 * on the base URL), so no case ever touches the real network.
 */
class LlmClientThinkingTest {

    private var lastBody: JSONObject? = null

    private val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
        createContext("/") { ex ->
            lastBody = JSONObject(ex.requestBody.bufferedReader().readText())
            val b = """{"choices":[{"message":{"role":"assistant","content":"hi"}}],"usage":{}}""".toByteArray()
            ex.responseHeaders.add("Content-Type", "application/json")
            ex.sendResponseHeaders(200, b.size.toLong()); ex.responseBody.use { it.write(b) }
        }
        start()
    }

    @After fun stop() = server.stop(0)

    /** [segment] stands in for the provider host; [model] is the model asked for. */
    private fun call(segment: String, model: String): JSONObject {
        val c = LlmClient("http://127.0.0.1:${server.address.port}/$segment", "k", model, true, "high")
        c.complete(listOf(JSONObject().put("role", "user").put("content", "hi")), JSONArray())
        return lastBody ?: throw AssertionError("no request captured")
    }

    @Test fun openAiGetsNeitherField() {
        val body = call("openai", "gpt-4o-mini")
        assertFalse(body.has("thinking"))
        assertFalse(body.has("reasoning_effort"))
        assertEquals("gpt-4o-mini", body.getString("model"))
    }

    @Test fun geminiGetsNeitherField() {
        val body = call("gemini", "gemini-3.8-flash")
        assertFalse(body.has("thinking"))
        assertFalse(body.has("reasoning_effort"))
    }

    @Test fun metaGetsNeitherField() {
        val body = call("meta", "muse-spark-1.3")
        assertFalse(body.has("thinking"))
        assertFalse(body.has("reasoning_effort"))
    }

    @Test fun llamaGetsNeitherField() {
        val body = call("llama", "Llama-4-Maverick-17B-128E-Instruct-FP8")
        assertFalse(body.has("thinking"))
        assertFalse(body.has("reasoning_effort"))
    }

    @Test fun deepSeekKeepsBothFields() {
        val body = call("deepseek", "deepseek-flash")
        assertEquals("enabled", body.getJSONObject("thinking").getString("type"))
        assertEquals("high", body.getString("reasoning_effort"))
    }

    @Test fun deepSeekModelOnAnotherHostKeepsBothFields() {
        // CodeFlying's gateway: its own host, DeepSeek's model.
        val body = call("cf", "deepseek-flash")
        assertEquals("enabled", body.getJSONObject("thinking").getString("type"))
        assertEquals("high", body.getString("reasoning_effort"))
    }

    @Test fun nonDeepSeekHostAndModelGetsNeitherField() {
        val body = call("other", "some-model")
        assertFalse(body.has("thinking"))
        assertFalse(body.has("reasoning_effort"))
    }

    @Test fun supportsThinkingMatchesHostOrModel() {
        assertTrue(LlmClient.supportsThinking("https://api.deepseek.com", "other-model"))
        assertTrue(LlmClient.supportsThinking("https://example.com", "deepseek-flash"))
        assertTrue(LlmClient.supportsThinking("https://API.DEEPSEEK.COM", "DeepSeek-Chat"))
        assertFalse(LlmClient.supportsThinking("https://api.openai.com/v1", "gpt-4o-mini"))
        assertFalse(LlmClient.supportsThinking("https://api.llama.com/compat/v1", "Llama-4-Maverick-17B-128E-Instruct-FP8"))
    }
}
