"""
BONZO API Gateway
FastAPI backend serving all Bonzo projects
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import redis
import psycopg2
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="BONZO API Gateway",
    description="Unified API for Bonzo ecosystem projects",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS[0] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connections
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bonzo:bonzo_dev_2026@postgres:5432/bonzo_main")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Redis client (lazy initialization)
redis_client: Optional[redis.Redis] = None

def get_redis():
    """Get Redis client with lazy initialization"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}")
            redis_client = None
    return redis_client


# Pydantic Models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    environment: str


class PUMOSyncRequest(BaseModel):
    shop_url: str
    api_key: str
    sync_products: bool = True
    sync_orders: bool = False


# Root Endpoint
@app.get("/")
async def root():
    """API Gateway root endpoint"""
    return {
        "service": "BONZO API Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Health Check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all services"""
    services = {}
    
    # Check Redis
    try:
        r = get_redis()
        if r:
            r.ping()
            services["redis"] = "healthy"
        else:
            services["redis"] = "unavailable"
    except Exception as e:
        services["redis"] = f"error: {str(e)[:50]}"
    
    # Check PostgreSQL
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        services["postgres"] = "healthy"
    except Exception as e:
        services["postgres"] = f"error: {str(e)[:50]}"
    
    overall_status = "healthy" if all(
        v == "healthy" for v in services.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        services=services,
        environment=os.getenv("ENVIRONMENT", "development")
    )


# PUMO Integration Endpoints
@app.get("/v1/pumo/health")
async def pumo_health():
    """PUMO integration health check"""
    return {
        "service": "PUMO Integration",
        "status": "ready",
        "db": "postgres",
        "cache": "redis",
        "shop_url": os.getenv("IDOSELL_SHOP_URL", "not_configured")
    }


@app.post("/v1/pumo/sync")
async def pumo_sync(request: PUMOSyncRequest):
    """
    Sync products and orders from IdoSell
    TODO: Implement actual IdoSell API integration
    """
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    # Store sync request in Redis for processing
    sync_id = f"pumo_sync_{datetime.utcnow().timestamp()}"
    r.setex(
        sync_id,
        3600,  # 1 hour expiry
        str(request.dict())
    )
    
    return {
        "sync_id": sync_id,
        "status": "queued",
        "message": "Sync request queued for processing"
    }


# Cache Management
@app.get("/v1/cache/stats")
async def cache_stats():
    """Get Redis cache statistics"""
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    info = r.info("stats")
    return {
        "total_connections": info.get("total_connections_received"),
        "total_commands": info.get("total_commands_processed"),
        "keyspace_hits": info.get("keyspace_hits"),
        "keyspace_misses": info.get("keyspace_misses"),
        "used_memory_human": r.info("memory").get("used_memory_human")
    }


@app.delete("/v1/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear cache keys matching pattern"""
    r = get_redis()
    if not r:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    keys = r.keys(pattern)
    if keys:
        r.delete(*keys)
    
    return {
        "cleared": len(keys),
        "pattern": pattern
    }


# Error Handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )



# ============================================================================
# MCP (Model Context Protocol) Endpoints
# ============================================================================

class MCPTool(BaseModel):
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPExecuteRequest(BaseModel):
    """MCP tool execution request"""
    name: str
    arguments: Dict[str, Any]


class MCPContentItem(BaseModel):
    """MCP response content item"""
    type: str  # "text", "json", "markdown"
    data: Any


class MCPExecuteResponse(BaseModel):
    """MCP tool execution response"""
    content: list[MCPContentItem]


# MCP Tools Registry
MCP_TOOLS: list[MCPTool] = [
    MCPTool(
        name="workspace_search",
        description="Search through workspace files and content",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "project": {
                    "type": "string",
                    "description": "Optional project filter"
                }
            },
            "required": ["query"]
        }
    ),
    MCPTool(
        name="workspace_stats",
        description="Get workspace statistics (projects, files, indexed count)",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    ),
    MCPTool(
        name="docker_status",
        description="Get Docker containers status",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    )
]


@app.get("/mcp/tools")
async def mcp_list_tools():
    """
    List all available MCP tools
    Spec: https://spec.modelcontextprotocol.io/specification/server/tools/
    """
    return {
        "tools": [tool.dict() for tool in MCP_TOOLS]
    }


@app.post("/mcp/tools/execute", response_model=MCPExecuteResponse)
async def mcp_execute_tool(request: MCPExecuteRequest):
    """
    Execute an MCP tool
    Spec: https://spec.modelcontextprotocol.io/specification/server/tools/
    
    Response format uses MCP content array for extensibility
    """
    import httpx
    
    # Find tool
    tool = next((t for t in MCP_TOOLS if t.name == request.name), None)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")
    
    # Execute tool by proxying to dashboard backend
    DASHBOARD_BACKEND = os.getenv("DASHBOARD_BACKEND_URL", "http://localhost:3880")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if request.name == "workspace_search":
                # Map to /api/search
                query = request.arguments.get("query", "")
                project = request.arguments.get("project")
                params = {"q": query}
                if project:
                    params["project"] = project
                    
                resp = await client.get(f"{DASHBOARD_BACKEND}/api/search", params=params)
                resp.raise_for_status()
                data = resp.json()
                
                return MCPExecuteResponse(
                    content=[
                        MCPContentItem(type="json", data=data)
                    ]
                )
                
            elif request.name == "workspace_stats":
                resp = await client.get(f"{DASHBOARD_BACKEND}/api/stats")
                resp.raise_for_status()
                data = resp.json()
                
                return MCPExecuteResponse(
                    content=[
                        MCPContentItem(type="json", data=data)
                    ]
                )
                
            elif request.name == "docker_status":
                resp = await client.get(f"{DASHBOARD_BACKEND}/api/docker")
                resp.raise_for_status()
                data = resp.json()
                
                return MCPExecuteResponse(
                    content=[
                        MCPContentItem(type="json", data=data)
                    ]
                )
                
            else:
                raise HTTPException(status_code=501, detail=f"Tool '{request.name}' not implemented")
                
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Dashboard backend error: {str(e)}")


@app.get("/mcp/resources")
async def mcp_list_resources():
    """
    List available MCP resources
    Spec: https://spec.modelcontextprotocol.io/specification/server/resources/
    """
    return {
        "resources": [
            {
                "uri": "workspace://stats",
                "name": "Workspace Statistics",
                "description": "Real-time workspace metrics",
                "mimeType": "application/json"
            },
            {
                "uri": "workspace://projects",
                "name": "Projects List",
                "description": "All workspace projects",
                "mimeType": "application/json"
            }
        ]
    }


@app.get("/mcp/prompts")
async def mcp_list_prompts():
    """
    List available MCP prompts
    Spec: https://spec.modelcontextprotocol.io/specification/server/prompts/
    """
    return {
        "prompts": [
            {
                "name": "code_review",
                "description": "Generate code review for workspace files",
                "arguments": [
                    {
                        "name": "file_path",
                        "description": "Path to file",
                        "required": True
                    }
                ]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )

