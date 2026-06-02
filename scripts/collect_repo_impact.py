"""Collect public GitHub and Crossref impact metrics for this repository.

The script uses only Python's standard library so it can run before project
dependencies are installed.
"""

from __future__ import annotations

import json
import urllib.request
from collections import Counter


REPOSITORY = "pritom007/Network-Traffic-Classification"
DOI = "10.1080/09540091.2020.1870437"
GOOGLE_SCHOLAR_PROFILE = "https://scholar.google.com/citations?view_op=list_works&hl=en&user=8CqVGlkAAAAJ"


def fetch_json(url: str, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def count_by_year(items: list[dict], key: str) -> Counter:
    return Counter(item[key][:4] for item in items)


def print_table(title: str, rows: list[tuple[str, int]]) -> None:
    print(f"\n{title}")
    print("| Year | Count |")
    print("| --- | ---: |")
    for year, count in rows:
        print(f"| {year} | {count} |")


def main() -> None:
    repo = fetch_json(f"https://api.github.com/repos/{REPOSITORY}")
    stars = fetch_json(
        f"https://api.github.com/repos/{REPOSITORY}/stargazers?per_page=100",
        headers={"Accept": "application/vnd.github.star+json"},
    )
    forks = fetch_json(f"https://api.github.com/repos/{REPOSITORY}/forks?per_page=100&sort=oldest")
    crossref = fetch_json(f"https://api.crossref.org/works/{DOI}")["message"]

    print("Repository impact snapshot")
    print(f"Stars: {repo['stargazers_count']}")
    print(f"Forks: {repo['forks_count']}")
    print(f"Watchers/subscribers: {repo['subscribers_count']}")
    print(f"Open issues: {repo['open_issues_count']}")
    print(f"Crossref cited-by count: {crossref['is-referenced-by-count']}")
    print(f"Crossref references: {len(crossref.get('reference', []))}")
    print(f"Google Scholar profile: {GOOGLE_SCHOLAR_PROFILE}")
    print("Google Scholar counts should be refreshed manually; Scholar does not provide a stable public API.")

    star_years = sorted(count_by_year(stars, "starred_at").items())
    fork_years = sorted(count_by_year(forks, "created_at").items())
    print_table("Stars by year", star_years)
    print_table("Forks by year", fork_years)

    if stars:
        first_star = min(stars, key=lambda item: item["starred_at"])
        latest_star = max(stars, key=lambda item: item["starred_at"])
        print("\nStar milestones")
        print(f"First: {first_star['user']['login']} at {first_star['starred_at']}")
        print(f"Latest: {latest_star['user']['login']} at {latest_star['starred_at']}")

    if forks:
        first_fork = min(forks, key=lambda item: item["created_at"])
        latest_fork = max(forks, key=lambda item: item["created_at"])
        print("\nFork milestones")
        print(f"First: {first_fork['full_name']} at {first_fork['created_at']}")
        print(f"Latest: {latest_fork['full_name']} at {latest_fork['created_at']}")


if __name__ == "__main__":
    main()
