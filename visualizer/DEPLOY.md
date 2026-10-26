# Memory Travel — Deployment & Production Guide

This guide covers building, configuring, and deploying the **Memory Travel** frontend visualizer and its companion **HARS Trace / Graph-Stats Server** using Docker and Kubernetes.

---

## 1. Architecture & Proxy Routes

Memory Travel is a browser visualizer (React 19, TypeScript, Three.js WebGL2 with Canvas2D fallback) served by a minimal non-root Nginx container. It connects to two backend services:

1. **Memory MCP Bridge** (Streamable HTTP MCP, typically port `8096` or `8099`):
   - Handles read-only `memory_recall` and `memory_status` tool calls.
2. **Trace / Graph-Stats Server** (`hars_memory.trace`, port `8098`):
   - Streams live recall telemetry via Server-Sent Events (SSE).
   - Exports the connected graph and 3D UMAP coordinates (`/stats`, `/graph`, `/coords`).

### Proxy Routing Table

The Nginx reverse proxy routes all requests same-origin to prevent browser CORS issues, supporting both sub-path (`/travel/`) and root (`/`) deployments:

| Client Browser Path | Proxied Target | Protocol | Description |
|---|---|---|---|
| `/travel/api/memory`<br>`/api/memory` | `http://${BACKEND_HOST}:${BACKEND_MEMORY_PORT}/mcp` | HTTP/1.1 POST | MCP Streamable HTTP bridge (`memory_recall`). |
| `/travel/api/trace/*`<br>`/api/trace/*` | `http://${BACKEND_TRACE_HOST}:${BACKEND_TRACE_PORT}/*` | HTTP/1.1 SSE (GET) | Live recall stage telemetry. Buffering is disabled. |
| `/travel/api/graph-stats/*`<br>`/api/graph-stats/*` | `http://${BACKEND_TRACE_HOST}:${BACKEND_TRACE_PORT}/*` | HTTP/1.1 GET | Read-only cosmos export: `/stats`, `/graph`, `/coords`. |
| `/healthz` | Local Nginx | HTTP GET | Returns `200 ok` for Kubernetes liveness & readiness probes. |
| `/travel/*` or `/*` | `/usr/share/nginx/html` | HTTP GET | Serves SPA static assets and `index.html`. |

### Auth Token Handling

Authentication for the upstream memory service is managed automatically:
- **Client Header Passthrough:** If the incoming client request contains an `Authorization: Bearer <token>` header, Nginx forwards it directly to the upstream backend.
- **Server Default Token:** If the client does not provide an `Authorization` header, Nginx injects `Authorization: Bearer ${AUTH_TOKEN}` using the token configured via the `AUTH_TOKEN` (or `HARS_MEMORY_ACCESS_TOKEN`) environment variable or Kubernetes Secret.
- **Header Suppression:** If neither the client nor the environment provides a token, the `Authorization` header is suppressed completely.

---

## 2. Docker: Build & Run Locally

### Build the Multi-Stage Image

The Dockerfile uses a clean two-stage build:
- **Stage 1 (Builder):** Uses `node:22-alpine`, installs dependencies with `npm ci`, runs the no-emoji prebuild validation gate, and compiles Vite + TypeScript.
- **Stage 2 (Runtime):** Uses minimal `nginx:alpine` configured for non-root execution (UID 1001) with dynamic DNS resolution and template variable substitution.

```bash
# Build with default subpath base (/travel/)
docker build -t memory-travel:latest .

# Or build for root domain hosting (/)
docker build --build-arg VITE_BASE=/ -t memory-travel:latest .
```

### Run Locally via Docker

#### 1. Demo Mode (Offline Snapshot)
Starts without upstream dependencies. Users can click **[Open demo data]** to view the baked snapshot:
```bash
docker run -d --name memory-travel -p 8080:8080 memory-travel:latest
```
Visit http://localhost:8080/travel/ (or http://localhost:8080/ if built with `VITE_BASE=/`).

#### 2. Connected Mode (Live Memory Backend)
Point the container at an existing MCP bridge and trace service:
```bash
docker run -d --name memory-travel -p 8080:8080 \
  -e BACKEND_HOST="memory-host" \
  -e BACKEND_MEMORY_PORT="8099" \
  -e BACKEND_TRACE_PORT="8098" \
  -e AUTH_TOKEN="optional-secret-token" \
  memory-travel:latest
```

#### 3. Docker Compose (Full Stack with Trace Server)
```yaml
version: "3.8"
services:
  memory-travel:
    build: .
    ports:
      - "8080:8080"
    environment:
      BACKEND_HOST: trace-server
      BACKEND_MEMORY_PORT: 8096
      BACKEND_TRACE_PORT: 8098
      AUTH_TOKEN: ""
    depends_on:
      - trace-server

  trace-server:
    image: hars-longterm-memory:latest
    command: ["python3", "-m", "hars_memory.trace"]
    environment:
      HARS_MEMORY_INDEX_DIR: /data/index
      HARS_MEMORY_TRACE_HTTP_HOST: "0.0.0.0"
      HARS_MEMORY_TRACE_HTTP_PORT: "8098"
    volumes:
      - /path/to/hars/index:/data/index:ro
    ports:
      - "8098:8098"
```

---

## 3. Kubernetes Deployment

Production-ready Kubernetes manifests are provided under `deploy/k8s/`.

### Directory Layout

```
deploy/k8s/
├── kustomization.yaml           # 1-command deployment manifest
├── deployment.yaml              # Frontend deployment (non-root UID 1001, probes, limits)
├── service.yaml                 # ClusterIP service on port 80
├── ingress.yaml                 # Ingress with SSE, TLS, and OAuth annotations
├── trace-server-deployment.yaml # Trace server deployment + service + PVC
├── configmap.yaml               # Backend host/port configuration
├── secret.yaml                  # Upstream auth token secret
└── values.yaml                  # Declarative Helm-ready values reference
```

### 1-Command Deployment with Kustomize

To deploy all components to the `memory-travel` namespace:

```bash
# Preview manifests
kubectl kustomize deploy/k8s/

# Apply to cluster
kubectl apply -k deploy/k8s/
```

### Deploying to a Custom Namespace or Host

To customize without editing files directly:

```bash
# Deploy to a different namespace
kubectl apply -k deploy/k8s/ -n custom-namespace

# Or set custom namespace and ingress host using Kustomize:
cd deploy/k8s
kustomize edit set namespace monitoring
sed -i '' 's/memory-travel.example.com/travel.my-company.com/g' ingress.yaml
kubectl apply -k .
```

### Configuration Details

#### ConfigMap (`deploy/k8s/configmap.yaml`)
Configures where the frontend Nginx proxies outgoing requests:
```yaml
data:
  BACKEND_HOST: "trace-server"      # Upstream service hostname
  BACKEND_MEMORY_PORT: "8096"       # MCP streamable HTTP port
  BACKEND_MEMORY_PATH: "/mcp"       # MCP endpoint path
  BACKEND_TRACE_HOST: "trace-server"# Trace service hostname
  BACKEND_TRACE_PORT: "8098"        # Trace HTTP / SSE port
```

#### Secret (`deploy/k8s/secret.yaml`)
Sets the fallback Bearer token for upstream requests:
```yaml
stringData:
  AUTH_TOKEN: "your-bearer-token-here"
```

#### Ingress (`deploy/k8s/ingress.yaml`)
Includes required annotations for Server-Sent Events (SSE) and optional OAuth/SSO:
- `nginx.ingress.kubernetes.io/proxy-buffering: "off"`: **Mandatory** for SSE stage events to stream in real time.
- `nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"`: Prevents long-lived trace streams from timing out.
- Annotations for `cert-manager` TLS and `oauth2-proxy` / SSO are provided as commented templates.

#### Trace Server (`deploy/k8s/trace-server-deployment.yaml`)
Runs the dedicated trace / graph-stats server:
- Executes `python3 -m hars_memory.trace` on port `8098`.
- Mounts the HARS index directory (`/data/index`) via `hars-memory-index-pvc` (ReadOnlyMany).
- Implements `/healthz` HTTP readiness and liveness probes.
- Enforces non-root securityContext (UID 10001).

---

## 4. Verification & Diagnostics

Once deployed, verify health across all components:

```bash
# 1. Check frontend container health
kubectl exec -it deployment/memory-travel -n memory-travel -- curl -s http://localhost:8080/healthz
# Expected: ok

# 2. Check trace server health
kubectl exec -it deployment/trace-server -n memory-travel -- curl -s http://localhost:8098/healthz
# Expected: {"ok": true, "service": "hars-longterm-memory", "trace": true}

# 3. Check graph export from frontend proxy
kubectl run curl-test --rm -it --image=curlimages/curl -- \
  curl -fsS http://memory-travel.memory-travel.svc.cluster.local/api/graph-stats/stats
```
