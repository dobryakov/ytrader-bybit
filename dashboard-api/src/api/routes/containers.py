"""Containers status endpoints."""

import subprocess
import json
import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ...config.logging import get_logger
from ...utils.tracing import get_or_create_trace_id

logger = get_logger(__name__)
router = APIRouter()

# Try to import docker SDK, fallback to subprocess if not available
try:
    import docker
    DOCKER_SDK_AVAILABLE = True
except ImportError:
    DOCKER_SDK_AVAILABLE = False
    logger.warning("docker SDK not available, will use subprocess")


def parse_docker_compose_ps(output: str) -> List[Dict[str, Any]]:
    """
    Parse docker compose ps output.
    
    Expected format:
    NAME                 IMAGE                COMMAND                  SERVICE              CREATED          STATUS                          PORTS
    ws-gateway           ytrader-ws-gateway   "uvicorn src.main:ap…"   ws-gateway           23 minutes ago   Up 39 seconds (healthy)        127.0.0.1:4400->4400/tcp
    """
    lines = output.strip().split('\n')
    if len(lines) < 2:
        return []
    
    # Skip header line
    containers = []
    for line in lines[1:]:
        if not line.strip():
            continue
        
        # Split by whitespace, but preserve quoted strings
        parts = line.split()
        if len(parts) < 6:
            continue
        
        # Extract fields
        name = parts[0]
        image = parts[1] if len(parts) > 1 else ''
        service = parts[3] if len(parts) > 3 else ''
        created = parts[4] if len(parts) > 4 else ''
        
        # Status is usually in format "Up X seconds (healthy)" or "Restarting (3) X seconds ago"
        status_parts = parts[5:] if len(parts) > 5 else []
        status_str = ' '.join(status_parts)
        
        # Extract health status
        health_status = None
        if '(healthy)' in status_str:
            health_status = 'healthy'
        elif '(unhealthy)' in status_str:
            health_status = 'unhealthy'
        elif '(health: starting)' in status_str:
            health_status = 'starting'
        elif 'Up' in status_str:
            health_status = 'running'
        elif 'Restarting' in status_str:
            health_status = 'restarting'
        elif 'Exited' in status_str:
            health_status = 'exited'
        else:
            health_status = 'unknown'
        
        # Extract ports if present
        ports = []
        if len(parts) > 6:
            for part in parts[6:]:
                if '->' in part or ':' in part:
                    ports.append(part)
        
        containers.append({
            'name': name,
            'image': image,
            'service': service,
            'created': created,
            'status': status_str,
            'health_status': health_status,
            'ports': ports,
        })
    
    return containers


def get_health_status_from_docker(container) -> str:
    """Extract health status from Docker container object."""
    if hasattr(container, 'attrs') and 'State' in container.attrs:
        health = container.attrs.get('State', {}).get('Health', {})
        if health:
            status = health.get('Status', '')
            if status == 'healthy':
                return 'healthy'
            elif status == 'unhealthy':
                return 'unhealthy'
            elif status == 'starting':
                return 'starting'
        
        state = container.attrs.get('State', {}).get('Status', '')
        if state == 'running':
            return 'running'
        elif state == 'restarting':
            return 'restarting'
        elif state == 'exited':
            return 'exited'
    
    # Fallback: check status string
    status_str = str(container.status).lower()
    if 'healthy' in status_str:
        return 'healthy'
    elif 'unhealthy' in status_str:
        return 'unhealthy'
    elif 'starting' in status_str:
        return 'starting'
    elif 'up' in status_str or 'running' in status_str:
        return 'running'
    elif 'restarting' in status_str:
        return 'restarting'
    elif 'exited' in status_str:
        return 'exited'
    
    return 'unknown'


@router.get("/containers")
async def get_containers_status():
    """Get Docker containers status."""
    trace_id = get_or_create_trace_id()
    logger.info("containers_status_request", trace_id=trace_id)

    try:
        containers = []
        
        # Try using Docker SDK first
        if DOCKER_SDK_AVAILABLE:
            try:
                # Try to connect to Docker daemon
                # First try socket, then HTTP
                client = None
                docker_socket = '/var/run/docker.sock'
                
                if os.path.exists(docker_socket):
                    try:
                        client = docker.DockerClient(base_url='unix://' + docker_socket)
                        # Test connection
                        client.ping()
                    except Exception as e:
                        logger.warning(
                            "containers_status_docker_socket_failed",
                            error=str(e),
                            trace_id=trace_id,
                        )
                        client = None
                
                if not client:
                    # Try HTTP connection (if Docker daemon exposes API)
                    docker_host = os.environ.get('DOCKER_HOST')
                    if docker_host:
                        try:
                            client = docker.DockerClient(base_url=docker_host)
                            client.ping()
                        except Exception as e:
                            logger.warning(
                                "containers_status_docker_http_failed",
                                error=str(e),
                                trace_id=trace_id,
                            )
                            client = None
                
                if not client:
                    raise Exception("Cannot connect to Docker daemon")
                
                # Get containers filtered by project label (only current project)
                project_name = os.environ.get('COMPOSE_PROJECT_NAME', 'ytrader')
                docker_containers = client.containers.list(
                    all=True,
                    filters={'label': f'com.docker.compose.project={project_name}'}
                )
                
                for container in docker_containers:
                    # Get container labels to find service name
                    labels = container.labels
                    service_name = labels.get('com.docker.compose.service', '')
                    
                    # Format ports
                    ports = []
                    if container.ports:
                        for port_info in container.ports.values():
                            if port_info:
                                for p in port_info:
                                    if 'HostPort' in p:
                                        ports.append(f"{p.get('HostIp', '0.0.0.0')}:{p['HostPort']}")
                    
                    health_status = get_health_status_from_docker(container)
                    
                    containers.append({
                        'name': container.name,
                        'image': container.image.tags[0] if container.image.tags else str(container.image),
                        'service': service_name,
                        'created': container.attrs.get('Created', ''),
                        'status': container.status,
                        'health_status': health_status,
                        'ports': ports,
                    })
                
                client.close()
                logger.info("containers_status_using_docker_sdk", count=len(containers), trace_id=trace_id)
                
            except Exception as docker_error:
                logger.warning(
                    "containers_status_docker_sdk_failed",
                    error=str(docker_error),
                    trace_id=trace_id,
                )
                # Fallback to subprocess
                pass
        
        # Fallback to subprocess if Docker SDK failed or not available
        if not containers:
            # Try docker compose ps command
            # Use current working directory or try to find project root
            project_dir = os.environ.get('COMPOSE_PROJECT_DIR')
            if not project_dir:
                # Try common locations or use current directory
                cwd = os.getcwd()
                # If we're in /app (container workdir), try to find project root
                if cwd == '/app':
                    # Look for docker-compose.yml in parent directories
                    for parent in ['/home/ubuntu/ytrader', '/app', '/']:
                        if os.path.exists(os.path.join(parent, 'docker-compose.yml')):
                            project_dir = parent
                            break
                if not project_dir:
                    project_dir = cwd
            
            result = subprocess.run(
                ['docker', 'compose', 'ps', '--format', 'json'],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=project_dir,
            )
            
            if result.returncode != 0:
                # Fallback to plain docker compose ps
                result = subprocess.run(
                    ['docker', 'compose', 'ps'],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=project_dir,
                )
                
                if result.returncode != 0:
                    logger.error(
                        "containers_status_command_failed",
                        error=result.stderr,
                        trace_id=trace_id,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to get containers status: {result.stderr}"
                    )
                
                # Parse plain text output
                containers = parse_docker_compose_ps(result.stdout)
            else:
                # Parse JSON output
                try:
                    json_lines = [line for line in result.stdout.strip().split('\n') if line.strip()]
                    containers = []
                    for line in json_lines:
                        container_data = json.loads(line)
                        # Extract health status from State
                        state = container_data.get('State', '')
                        health = container_data.get('Health', '')
                        
                        if health == 'healthy':
                            health_status = 'healthy'
                        elif health == 'unhealthy':
                            health_status = 'unhealthy'
                        elif 'starting' in state.lower() or 'health: starting' in state:
                            health_status = 'starting'
                        elif 'up' in state.lower():
                            health_status = 'running'
                        elif 'restarting' in state.lower():
                            health_status = 'restarting'
                        elif 'exited' in state.lower():
                            health_status = 'exited'
                        else:
                            health_status = 'unknown'
                        
                        ports_str = container_data.get('Ports', '')
                        ports = [p.strip() for p in ports_str.split(',') if p.strip()] if ports_str else []
                        
                        containers.append({
                            'name': container_data.get('Name', ''),
                            'image': container_data.get('Image', ''),
                            'service': container_data.get('Service', ''),
                            'created': container_data.get('Created', ''),
                            'status': state,
                            'health_status': health_status,
                            'ports': ports,
                        })
                except json.JSONDecodeError as e:
                    logger.error(
                        "containers_status_json_parse_failed",
                        error=str(e),
                        trace_id=trace_id,
                    )
                    # Fallback to plain text parsing
                    containers = parse_docker_compose_ps(result.stdout)
            
            logger.info("containers_status_using_subprocess", count=len(containers), trace_id=trace_id)

        logger.info(
            "containers_status_completed",
            count=len(containers),
            trace_id=trace_id,
        )

        return JSONResponse(
            status_code=200,
            content={
                "containers": containers,
                "count": len(containers),
            },
        )

    except subprocess.TimeoutExpired:
        logger.error("containers_status_timeout", trace_id=trace_id)
        raise HTTPException(
            status_code=504,
            detail="Timeout while getting containers status"
        )
    except Exception as e:
        logger.error(
            "containers_status_failed",
            error=str(e),
            trace_id=trace_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve containers status: {str(e)}"
        )

