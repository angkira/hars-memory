# =============================================================================
# Stage 1: Build the React 19 / Vite frontend
# =============================================================================
FROM node:22-alpine AS builder

WORKDIR /build

# Copy dependency manifests first to leverage Docker layer caching
COPY app/package.json app/package-lock.json ./app/

WORKDIR /build/app
RUN npm ci

# Copy scripts (needed for the no-emoji prebuild validation gate) and source
WORKDIR /build
COPY scripts/ ./scripts/
COPY app/ ./app/

# Build arguments for base path and optional baked fallback URLs
ARG VITE_BASE=/travel/
ARG MEMORY_BRIDGE_URL=
ARG MEMORY_TRACE_URL=

ENV VITE_BASE=${VITE_BASE} \
    MEMORY_BRIDGE_URL=${MEMORY_BRIDGE_URL} \
    MEMORY_TRACE_URL=${MEMORY_TRACE_URL}

WORKDIR /build/app
RUN npm run build

# =============================================================================
# Stage 2: Minimal non-root Nginx production runtime
# =============================================================================
FROM nginx:alpine

# Create writable temp and cache directories for non-root execution (UID 1001)
RUN mkdir -p /var/cache/nginx /etc/nginx/conf.d /etc/nginx/templates \
             /tmp/client_temp /tmp/proxy_temp_path /tmp/fastcgi_temp /tmp/uwsgi_temp /tmp/scgi_temp \
    && touch /tmp/nginx.pid \
    && chown -R 1001:0 /var/cache/nginx /etc/nginx /usr/share/nginx/html /tmp /tmp/nginx.pid \
    && chmod -R g+rwX /var/cache/nginx /etc/nginx /tmp

# Copy compiled static assets from builder stage
COPY --from=builder /build/app/dist /usr/share/nginx/html

# Copy Nginx main configuration and templated server definition
COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker-entrypoint.sh /docker-entrypoint.sh

# Pre-render default configuration and secure permissions
RUN chmod +x /docker-entrypoint.sh \
    && /docker-entrypoint.sh true \
    && chown -R 1001:0 /etc/nginx/conf.d /usr/share/nginx/html

USER 1001
EXPOSE 8080

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
