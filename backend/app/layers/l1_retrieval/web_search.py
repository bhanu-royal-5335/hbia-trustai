"""
Layer 1a: Web Search Engine
Fetches live online data sources using DuckDuckGo HTML, Wikipedia API (Search + Extracts),
Tavily, Serper, or Google APIs for live real-time knowledge.
"""
import re
import urllib.parse
from typing import List, Optional
import httpx
from app.core.config import settings
from app.layers.l1_retrieval.vector_store import RetrievedChunk
import structlog

logger = structlog.get_logger()


class WebSearchEngine:
    """Live online search engine that retrieves real-time web results."""

    def __init__(self):
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def search(self, query: str, num_results: int = 6) -> List[RetrievedChunk]:
        """Perform web search across multiple sources and return List[RetrievedChunk]."""
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        logger.info("web_search_started", query=clean_query)
        results: List[RetrievedChunk] = []

        # 1. Try Serper API if key is present
        serper_key = getattr(settings, "serper_api_key", None) or ""
        if serper_key:
            try:
                serper_res = await self._search_serper(clean_query, serper_key, num_results)
                if serper_res:
                    results.extend(serper_res)
                    logger.info("web_search_success", count=len(serper_res), engine="serper")
            except Exception as e:
                logger.warning("serper_search_failed", error=str(e))

        # 2. Try Tavily API if key is present
        tavily_key = getattr(settings, "tavily_api_key", None) or ""
        if tavily_key:
            try:
                tavily_res = await self._search_tavily(clean_query, tavily_key, num_results)
                if tavily_res:
                    results.extend(tavily_res)
                    logger.info("web_search_success", count=len(tavily_res), engine="tavily")
            except Exception as e:
                logger.warning("tavily_search_failed", error=str(e))

        # 3. DuckDuckGo HTML Live Web Search (Zero API Key needed)
        try:
            ddg_res = await self._search_duckduckgo(clean_query, num_results)
            if ddg_res:
                results.extend(ddg_res)
                logger.info("web_search_success", count=len(ddg_res), engine="duckduckgo")
        except Exception as e:
            logger.warning("duckduckgo_search_failed", error=str(e))

        # 4. Wikipedia API (Search + Article Intro Extracts)
        try:
            wiki_res = await self._search_wikipedia(clean_query, num_results)
            if wiki_res:
                results.extend(wiki_res)
                logger.info("web_search_success", count=len(wiki_res), engine="wikipedia")
        except Exception as e:
            logger.warning("wikipedia_search_failed", error=str(e))

        # Deduplicate results by content snippet
        seen_snippets = set()
        unique_results = []
        for r in results:
            short_sig = r.content[:80].lower()
            if short_sig not in seen_snippets:
                seen_snippets.add(short_sig)
                unique_results.append(r)

        return unique_results[:num_results]

    async def _search_duckduckgo(self, query: str, num_results: int) -> List[RetrievedChunk]:
        """Scrape DuckDuckGo HTML search for live real-time web results."""
        results = []
        async with httpx.AsyncClient(headers=self.headers, timeout=4.0, follow_redirects=True) as client:
            # POST request to DDG HTML
            resp = await client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
            )
            if resp.status_code != 200:
                return []

            html = resp.text
            # Match results: snippet & title & url
            items = re.findall(
                r'<a[^>]+class="result__url"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
                r'class="result__snippet"[^>]*>(.*?)</a>',
                html,
                re.DOTALL,
            )

            for i, (link, title_raw, snippet_raw) in enumerate(items[:num_results]):
                title = re.sub(r"<[^>]+>", "", title_raw).strip()
                snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
                snippet = (
                    snippet.replace("&quot;", '"')
                    .replace("&amp;", "&")
                    .replace("&#x27;", "'")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                )
                title = (
                    title.replace("&quot;", '"')
                    .replace("&amp;", "&")
                    .replace("&#x27;", "'")
                )

                # Decode DDG redirect URL
                actual_url = link
                if "uddg=" in link:
                    match = re.search(r"uddg=([^&]+)", link)
                    if match:
                        actual_url = urllib.parse.unquote(match.group(1))

                domain = urllib.parse.urlparse(actual_url).netloc or "web"

                if snippet and len(snippet) > 15:
                    results.append(
                        RetrievedChunk(
                            content=f"Live Web Search [{title}]: {snippet}",
                            doc_id=f"ddg_{i+1}",
                            filename=f"Web: {domain}",
                            chunk_index=i,
                            relevance_score=round(0.95 - (i * 0.03), 2),
                            metadata={
                                "url": actual_url,
                                "domain": domain,
                                "is_web": True,
                                "filename": f"Web: {domain}",
                                "title": title,
                            },
                        )
                    )
        return results

    async def _search_wikipedia(self, query: str, num_results: int) -> List[RetrievedChunk]:
        """Search Wikipedia API for article intros & extracts."""
        results = []
        wiki_headers = {"User-Agent": "HBIA-TrustAI/1.0 (Research Assistant)"}

        # Clean query to get key entities (e.g. remove question words)
        clean_terms = re.sub(r'\b(who|what|where|when|why|how|is|the|current|working|as|of)\b', '', query, flags=re.IGNORECASE).strip()
        search_query = clean_terms if len(clean_terms) > 3 else query

        async with httpx.AsyncClient(headers=wiki_headers, timeout=4.0) as client:
            # 1. Search for titles
            params = {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "format": "json",
                "utf8": 1,
                "srlimit": num_results,
            }
            res = await client.get("https://en.wikipedia.org/w/api.php", params=params)
            if res.status_code != 200:
                return []

            data = res.json()
            search_items = data.get("query", {}).get("search", [])
            if not search_items:
                return []

            titles = [item.get("title", "") for item in search_items if item.get("title")]

            # 2. Fetch full extracts for top matched Wikipedia titles
            titles_str = "|".join(titles[:4])
            extract_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": titles_str,
                "format": "json",
                "utf8": 1,
            }
            ext_res = await client.get("https://en.wikipedia.org/w/api.php", params=extract_params)
            extracts_map = {}
            if ext_res.status_code == 200:
                pages = ext_res.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    p_title = page.get("title", "")
                    p_extract = page.get("extract", "")
                    if p_title and p_extract:
                        extracts_map[p_title] = p_extract

            # Build RetrievedChunk objects
            for i, item in enumerate(search_items):
                title = item.get("title", "")
                snippet_raw = item.get("snippet", "")
                snippet = re.sub(r"<[^>]+>", "", snippet_raw).strip()
                snippet = snippet.replace("&quot;", '"').replace("&amp;", "&").replace("&#x27;", "'")

                # Prefer full page intro extract if available
                full_extract = extracts_map.get(title, "")
                content_text = full_extract if (full_extract and len(full_extract) > len(snippet)) else snippet

                url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"

                if content_text and len(content_text) > 15:
                    results.append(
                        RetrievedChunk(
                            content=f"Online Knowledge [{title}]: {content_text}",
                            doc_id=f"wiki_{i+1}",
                            filename=f"Web: wikipedia.org ({title})",
                            chunk_index=i,
                            relevance_score=round(0.92 - (i * 0.04), 2),
                            metadata={
                                "url": url,
                                "domain": "wikipedia.org",
                                "is_web": True,
                                "filename": f"Web: wikipedia.org ({title})",
                                "title": title,
                            },
                        )
                    )

        return results

    async def _search_serper(self, query: str, api_key: str, num_results: int) -> List[RetrievedChunk]:
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query, "num": num_results}
        async with httpx.AsyncClient(headers=headers, timeout=6.0) as client:
            res = await client.post("https://google.serper.dev/search", json=payload)
            res.raise_for_status()
            data = res.json()
            organic = data.get("organic", [])
            results = []
            for i, item in enumerate(organic[:num_results]):
                snippet = item.get("snippet", "")
                url = item.get("link", "")
                title = item.get("title", "")
                domain = urllib.parse.urlparse(url).netloc or "google.com"
                if snippet:
                    results.append(
                        RetrievedChunk(
                            content=f"Web Result [{title}]: {snippet}",
                            doc_id=f"serper_{i+1}",
                            filename=f"Web: {domain}",
                            chunk_index=i,
                            relevance_score=round(0.95 - (i * 0.04), 2),
                            metadata={
                                "url": url,
                                "domain": domain,
                                "is_web": True,
                                "filename": f"Web: {domain}",
                                "title": title,
                            },
                        )
                    )
            return results

    async def _search_tavily(self, query: str, api_key: str, num_results: int) -> List[RetrievedChunk]:
        payload = {"api_key": api_key, "query": query, "max_results": num_results}
        async with httpx.AsyncClient(timeout=6.0) as client:
            res = await client.post("https://api.tavily.com/search", json=payload)
            res.raise_for_status()
            data = res.json()
            items = data.get("results", [])
            results = []
            for i, item in enumerate(items[:num_results]):
                snippet = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", "")
                domain = urllib.parse.urlparse(url).netloc or "tavily.com"
                if snippet:
                    results.append(
                        RetrievedChunk(
                            content=f"Web Result [{title}]: {snippet}",
                            doc_id=f"tavily_{i+1}",
                            filename=f"Web: {domain}",
                            chunk_index=i,
                            relevance_score=round(0.96 - (i * 0.04), 2),
                            metadata={
                                "url": url,
                                "domain": domain,
                                "is_web": True,
                                "filename": f"Web: {domain}",
                                "title": title,
                            },
                        )
                    )
            return results


web_search_engine = WebSearchEngine()
