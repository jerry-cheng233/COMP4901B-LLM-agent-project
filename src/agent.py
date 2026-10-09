"""
Search-Augmented LLM Agent for Question Answering.

This module implements an agent that combines DeepSeek LLM with Google Search (via Serper API)
to answer questions. The agent follows a ReAct-style loop:
1. Reason about the question
2. Decide whether to search or answer
3. If search: formulate query, retrieve results, and iterate
4. If answer: provide the final answer

Termination conditions:
- The model decides it has enough information to answer
- Maximum search steps reached
- Model explicitly outputs a final answer
"""

import os
import json
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI


# Configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SERPER_API_URL = "https://google.serper.dev/search"

# Default hyperparameters
DEFAULT_MAX_SEARCH_STEPS = 3
DEFAULT_NUM_SEARCH_RESULTS = 5
DEFAULT_MAX_TOKENS = 1024


def google_search(query: str, num_results: int = DEFAULT_NUM_SEARCH_RESULTS) -> List[Dict[str, str]]:
    """
    Perform Google search using Serper API.
    
    Args:
        query: The search query
        num_results: Number of results to return
        
    Returns:
        List of search results with title, snippet, and link
    """
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": num_results
    }
    
    try:
        response = requests.post(SERPER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        results = []
        # Extract organic search results
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def format_search_results(results: List[Dict[str, str]]) -> str:
    """Format search results into a readable string for the LLM."""
    if not results:
        return "No search results found."

    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] {r['title']}\n{r['snippet']}\nURL: {r['link']}")
    return "\n\n".join(formatted)


def browse_url(url: str, max_chars: int = 4000) -> str:
    """
    Browse a URL and extract the main text content.

    Args:
        url: The URL to browse
        max_chars: Maximum characters to return (to avoid token limits)

    Returns:
        Extracted text content from the page
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Try to use BeautifulSoup if available, otherwise use basic extraction
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            # Get text
            text = soup.get_text(separator='\n', strip=True)
        except ImportError:
            # Basic HTML tag removal if BeautifulSoup not available
            import re
            text = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # Truncate to max_chars
        if len(text) > max_chars:
            text = text[:max_chars] + "... [truncated]"

        return text if text else "Could not extract content from page."
    except Exception as e:
        return f"Error browsing URL: {str(e)}"


class SearchAgent:
    """
    Search-augmented LLM agent for question answering.
    
    The agent uses a ReAct-style loop to:
    1. Analyze the question
    2. Decide whether to search for more information or provide an answer
    3. If searching, formulate effective queries and process results
    4. Synthesize information from multiple searches to answer
    """
    
    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        serper_key: str = SERPER_API_KEY,
        model: str = "deepseek-chat",
        max_search_steps: int = DEFAULT_MAX_SEARCH_STEPS,
        num_search_results: int = DEFAULT_NUM_SEARCH_RESULTS,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        global SERPER_API_KEY
        SERPER_API_KEY = serper_key
        
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_search_steps = max_search_steps
        self.num_search_results = num_search_results
        self.max_tokens = max_tokens
        
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the agent."""
        return """You are a QA system that provides EXTREMELY SHORT answers. You have access to Google Search.

CRITICAL ANSWER FORMAT RULES:
- Give ONLY the minimal answer - just the name, date, number, or key phrase
- NO full sentences, NO explanations, NO context
- Examples of good answers: "December 1972", "Bobby Scott", "one", "James I", "54 Mbps"
- Examples of BAD answers: "The answer is December 1972", "He was written by Bobby Scott"

When you need to search, respond with:
ACTION: search
QUERY: <search query>

When you have the answer, respond with:
ACTION: answer
ANSWER: <SHORT answer - just the key fact, name, date, or number>

Examples:
Q: When was the last time anyone was on the moon?
ANSWER: December 1972

Q: Who wrote a certain song?
ANSWER: Bobby Scott

Q: How many seasons of a show are there?
ANSWER: one"""

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract action and content."""
        response = response.strip()

        # Check for answer action - use the LAST occurrence of ANSWER:
        # This handles cases where the model generates multiple actions in one response
        response_lower = response.lower()

        if "answer:" in response_lower:
            # Find the last ANSWER: in the response
            last_answer_idx = response_lower.rfind("answer:")
            if last_answer_idx != -1:
                answer = response[last_answer_idx + 7:].strip()
                # Clean up - remove any trailing action markers
                if "\naction:" in answer.lower():
                    answer = answer.split("\n")[0].strip()
                return {"action": "answer", "content": answer}

        # Check for search action - use the FIRST occurrence
        if "action: search" in response_lower or "action:search" in response_lower:
            # Extract query after QUERY:
            if "query:" in response_lower:
                query_idx = response_lower.find("query:")
                query = response[query_idx + 6:].strip()
                # Take only first line
                query = query.split("\n")[0].strip()
                return {"action": "search", "content": query}
            else:
                return {"action": "search", "content": response}

        # Default: treat as direct answer
        return {"action": "answer", "content": response}

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the agent to answer a question.

        Args:
            question: The question to answer

        Returns:
            Dictionary containing:
            - final_answer: The agent's answer
            - trajectory: List of steps taken
            - total_search_steps: Number of searches performed
        """
        trajectory = {
            "question": question,
            "steps": [],
            "final_answer": "",
            "total_search_steps": 0
        }

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Question: {question}"}
        ]

        step_count = 0
        search_count = 0

        while search_count < self.max_search_steps:
            step_count += 1

            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens
            )

            assistant_message = response.choices[0].message.content
            parsed = self._parse_response(assistant_message)

            if parsed["action"] == "answer":
                trajectory["final_answer"] = parsed["content"]
                break

            elif parsed["action"] == "search":
                search_count += 1
                query = parsed["content"]

                # Perform search
                search_results = google_search(query, self.num_search_results)

                # Record step
                step_info = {
                    "step_number": search_count,
                    "action": "search",
                    "query": query,
                    "num_docs_requested": self.num_search_results,
                    "retrieved_documents": [
                        {"title": r["title"], "snippet": r["snippet"]}
                        for r in search_results
                    ]
                }
                trajectory["steps"].append(step_info)

                # Add to conversation
                messages.append({"role": "assistant", "content": assistant_message})

                formatted_results = format_search_results(search_results)
                messages.append({
                    "role": "user",
                    "content": f"Search results for '{query}':\n\n{formatted_results}\n\nBased on these results, either search again for more specific information or provide your final answer."
                })

        trajectory["total_search_steps"] = search_count

        # If we exhausted search steps without answering, get final answer
        if not trajectory["final_answer"]:
            messages.append({
                "role": "user",
                "content": "Please provide your final answer now based on the information gathered."
            })
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens
            )
            final = response.choices[0].message.content
            parsed = self._parse_response(final)
            trajectory["final_answer"] = parsed["content"]

        return trajectory


class DirectQA:
    """
    Direct QA without search - baseline model.
    Uses DeepSeek to answer questions directly without external search.
    """

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        model: str = "deepseek-chat",
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_tokens = max_tokens

    def answer(self, question: str) -> str:
        """
        Answer a question directly without search.

        Args:
            question: The question to answer

        Returns:
            The answer string
        """
        system_prompt = """You are a factual QA system. Give ONLY the direct answer with no extra words.

RULES:
- Answer with ONLY the specific fact, name, date, place, or number asked for
- NEVER start with "The", "It is", "Answer:", or any introduction
- NEVER explain or add context
- If multiple answers exist, give the most common/primary one

GOOD: "Paris" | "December 1972" | "James I" | "one" | "54 Mbps" | "Bobby Scott"
BAD: "The capital is Paris" | "It was December 1972" | "There is one season\""""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Answer in 1-5 words only: {question}"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_tokens
        )

        return response.choices[0].message.content.strip()


class BrowsingAgent:
    """
    Advanced search agent with browsing capabilities.

    This agent extends the basic SearchAgent by adding the ability to browse
    full web page contents, not just search snippets. The agent can:
    1. Search Google for relevant pages
    2. Browse specific URLs to get full content
    3. Answer based on comprehensive information
    """

    def __init__(
        self,
        api_key: str = DEEPSEEK_API_KEY,
        serper_key: str = SERPER_API_KEY,
        model: str = "deepseek-chat",
        max_steps: int = 4,
        num_search_results: int = DEFAULT_NUM_SEARCH_RESULTS,
        max_tokens: int = DEFAULT_MAX_TOKENS
    ):
        global SERPER_API_KEY
        SERPER_API_KEY = serper_key

        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = model
        self.max_steps = max_steps
        self.num_search_results = num_search_results
        self.max_tokens = max_tokens

    def _get_system_prompt(self) -> str:
        """Return the system prompt for the browsing agent."""
        return """You are a QA system with access to Google Search and web browsing. Give EXTREMELY SHORT answers.

You have TWO tools:
1. SEARCH - Find relevant web pages
2. BROWSE - Read full content of a specific URL

When you need to search, respond with:
ACTION: search
QUERY: <search query>

When you want to read a full web page (after seeing search results), respond with:
ACTION: browse
URL: <full url to browse>

When you have the answer, respond with:
ACTION: answer
ANSWER: <SHORT answer - just the key fact, name, date, or number>

IMPORTANT:
- Use BROWSE when search snippets don't have enough detail
- Give ONLY minimal answers (1-5 words): "December 1972", "James I", "one"
- NO full sentences in your final answer"""

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract action and content."""
        response = response.strip()
        response_lower = response.lower()

        # Check for answer
        if "answer:" in response_lower:
            last_answer_idx = response_lower.rfind("answer:")
            if last_answer_idx != -1:
                answer = response[last_answer_idx + 7:].strip()
                if "\naction:" in answer.lower():
                    answer = answer.split("\n")[0].strip()
                return {"action": "answer", "content": answer}

        # Check for browse action
        if "action: browse" in response_lower or "action:browse" in response_lower:
            if "url:" in response_lower:
                url_idx = response_lower.find("url:")
                url = response[url_idx + 4:].strip()
                url = url.split("\n")[0].strip()
                return {"action": "browse", "content": url}

        # Check for search action
        if "action: search" in response_lower or "action:search" in response_lower:
            if "query:" in response_lower:
                query_idx = response_lower.find("query:")
                query = response[query_idx + 6:].strip()
                query = query.split("\n")[0].strip()
                return {"action": "search", "content": query}

        return {"action": "answer", "content": response}

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the browsing agent to answer a question.

        Returns:
            Dictionary containing final_answer, trajectory, and step counts
        """
        trajectory = {
            "question": question,
            "steps": [],
            "final_answer": "",
            "total_search_steps": 0,
            "total_browse_steps": 0
        }

        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": f"Question: {question}"}
        ]

        step_count = 0
        search_count = 0
        browse_count = 0
        last_search_results = []

        while step_count < self.max_steps:
            step_count += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens
            )

            assistant_message = response.choices[0].message.content
            parsed = self._parse_response(assistant_message)

            if parsed["action"] == "answer":
                trajectory["final_answer"] = parsed["content"]
                break

            elif parsed["action"] == "search":
                search_count += 1
                query = parsed["content"]
                search_results = google_search(query, self.num_search_results)
                last_search_results = search_results

                step_info = {
                    "step_number": step_count,
                    "action": "search",
                    "query": query,
                    "retrieved_documents": [
                        {"title": r["title"], "snippet": r["snippet"], "url": r["link"]}
                        for r in search_results
                    ]
                }
                trajectory["steps"].append(step_info)

                messages.append({"role": "assistant", "content": assistant_message})
                formatted_results = format_search_results(search_results)
                messages.append({
                    "role": "user",
                    "content": f"Search results:\n\n{formatted_results}\n\nYou can BROWSE a URL for more details, SEARCH again, or provide your ANSWER."
                })

            elif parsed["action"] == "browse":
                browse_count += 1
                url = parsed["content"]
                page_content = browse_url(url)

                step_info = {
                    "step_number": step_count,
                    "action": "browse",
                    "url": url,
                    "content_preview": page_content[:500] + "..." if len(page_content) > 500 else page_content
                }
                trajectory["steps"].append(step_info)

                messages.append({"role": "assistant", "content": assistant_message})
                messages.append({
                    "role": "user",
                    "content": f"Page content from {url}:\n\n{page_content}\n\nBased on this, provide your ANSWER or continue searching."
                })

        trajectory["total_search_steps"] = search_count
        trajectory["total_browse_steps"] = browse_count

        # If no answer yet, force one
        if not trajectory["final_answer"]:
            messages.append({
                "role": "user",
                "content": "Provide your final answer now in 1-5 words."
            })
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens
            )
            final = response.choices[0].message.content
            parsed = self._parse_response(final)
            trajectory["final_answer"] = parsed["content"]

        return trajectory
