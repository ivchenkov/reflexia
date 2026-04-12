"""Web access tools."""

from __future__ import annotations

from typing import TypedDict

import requests
import trafilatura
from langchain.tools import ToolRuntime, tool

from reflexia.config import ExecutionContext


class SearchResult(TypedDict):
    """A single SearxNG search result."""

    title: str
    url: str


def search_web_impl(
    query: str,
    searxng_url: str,
    timeout_sec: int,
    max_results: int,
) -> list[SearchResult]:
    """Perform a web search via SearxNG and return top results."""

    response = requests.get(
        searxng_url,
        params={"q": query, "format": "json"},
        timeout=timeout_sec,
    )
    response.raise_for_status()
    data = response.json()

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
        }
        for result in data.get("results", [])[:max_results]
    ]


def read_webpage_impl(url: str, max_chars: int) -> str:
    """Fetch a web page and extract the main text content."""

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""

    text = trafilatura.extract(
        downloaded,
        url=url,
        favor_precision=True,
        include_comments=False,
        with_metadata=False,
    )
    if not text:
        return ""

    return f"URL: {url}\n\n{text[:max_chars]}"


@tool
def search_web(
    query: str,
    runtime: ToolRuntime[ExecutionContext],
) -> list[SearchResult]:
    """Find relevant web pages for a query."""

    ctx = runtime.context
    return search_web_impl(
        query=query,
        searxng_url=ctx.searxng_url,
        timeout_sec=ctx.web_timeout_sec,
        max_results=ctx.web_search_max_results,
    )


@tool
def read_webpage(
    url: str,
    runtime: ToolRuntime[ExecutionContext],
) -> str:
    """Read and extract the main text from a web page."""

    ctx = runtime.context
    return read_webpage_impl(
        url=url,
        max_chars=ctx.webpage_max_chars,
    )
