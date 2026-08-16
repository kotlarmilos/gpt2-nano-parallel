"""Deterministic top-k expert routing and all-to-all byte accounting."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TokenRoute:
    token: int
    source_rank: int
    expert: int
    destination_rank: int


def route_tokens(
    tokens: int,
    experts: int,
    world_size: int,
    *,
    top_k: int = 1,
) -> list[TokenRoute]:
    """Route tokens with a deterministic permutation standing in for gate scores."""

    if min(tokens, experts, world_size, top_k) <= 0:
        raise ValueError("tokens, experts, world_size, and top_k must be positive")
    if experts % world_size:
        raise ValueError("experts must be divisible by world_size")
    if top_k > experts:
        raise ValueError("top_k must not exceed experts")
    routes: list[TokenRoute] = []
    for token in range(tokens):
        source = token % world_size
        first = (token * 5 + token // max(1, experts)) % experts
        for choice in range(top_k):
            expert = (first + choice) % experts
            routes.append(TokenRoute(token, source, expert, expert % world_size))
    return routes


def routing_report(
    tokens: int,
    experts: int,
    world_size: int,
    *,
    hidden_size: int,
    top_k: int = 1,
    element_bytes: int = 2,
) -> dict[str, object]:
    if hidden_size <= 0 or element_bytes <= 0:
        raise ValueError("hidden_size and element_bytes must be positive")
    routes = route_tokens(tokens, experts, world_size, top_k=top_k)
    counts = [0] * experts
    remote_routes = 0
    for route in routes:
        counts[route.expert] += 1
        remote_routes += route.source_rank != route.destination_rank
    dispatch_bytes = remote_routes * hidden_size * element_bytes
    return {
        "evidence": "simulated",
        "tokens": tokens,
        "experts": experts,
        "world_size": world_size,
        "top_k": top_k,
        "expert_loads": counts,
        "remote_routes": remote_routes,
        "dispatch_bytes": dispatch_bytes,
        "combine_bytes": dispatch_bytes,
        "total_all_to_all_bytes": 2 * dispatch_bytes,
        "routes": [asdict(route) for route in routes],
    }
