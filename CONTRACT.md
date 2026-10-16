# The AgentCore Runtime contract (what your service must expose)

Amazon Bedrock AgentCore Runtime hosts any container that follows a small HTTP contract. If your service
follows it, it can be deployed to AgentCore without code changes. You do **not** need an AWS account for
this assignment. We only need the contract to hold locally.

| Requirement | Detail |
|---|---|
| Port | `8080`, listening on `0.0.0.0` |
| `POST /invocations` | The agent entrypoint. JSON in, JSON out. |
| `GET /ping` | Health check. Return `{"status": "Healthy"}` (or `HealthyBusy` while background work runs). |
| Session | AgentCore sends the session id in the header `X-Amzn-Bedrock-AgentCore-Runtime-Session-Id` (a UUID-like string, at least 33 characters). Keep short-term memory per session. |
| Container | A `Dockerfile`. AgentCore runs ARM64 images (`--platform linux/arm64`); if you can only test amd64 locally, say so in the README; we won't penalise it. |

## Request / response shape we will use

To keep everyone's eval sets and our review comparable, use this minimal shape. Add fields as you need.

```http
POST /invocations
Content-Type: application/json
X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: 3b1c9c2e-0f1d-4a53-9a9e-5d3f2b7e8c11-local

{"prompt": "Show me open tickets about VPN"}
```

```json
{
  "output": "There are 3 open VPN tickets: TCK-1001 ...",
  "tool_calls": [
    {"name": "search_tickets", "input": {"q": "VPN", "status": "open"}, "status": "ok", "duration_ms": 84}
  ],
  "pending_approval": null
}
```

When the agent needs confirmation before a write, return the proposed action instead of executing it:

```json
{
  "output": "I'll create a high-priority ticket 'Laptop won't boot' for Priya Raman. Shall I go ahead?",
  "tool_calls": [],
  "pending_approval": {"action": "create_ticket", "input": {"title": "Laptop won't boot", "priority": "high", "requester": "Priya Raman"}}
}
```

The user's answer arrives as the next `/invocations` call in the same session (`{"prompt": "yes, go ahead"}`).
Your service decides what counts as approval, what counts as a decline, and what to do if the user asks
something unrelated instead.

`tool_calls[].status` can be `ok`, `error`, `retried` or whatever you need. Make it honest: if the
ticketing API returned a 500 and you retried, we want to see that.

## Smoke test

```bash
docker build --platform linux/arm64 -t my-agent . && docker run -p 8080:8080 --env-file .env my-agent
curl http://localhost:8080/ping
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: 3b1c9c2e-0f1d-4a53-9a9e-5d3f2b7e8c11-local" \
  -d '{"prompt": "What is the status of TCK-1002?"}'
```
