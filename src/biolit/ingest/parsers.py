"""Parsers for NCBI E-utilities esearch / esummary / efetch payloads."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import date
from typing import Any

from biolit.ingest.models import PubMedDocument

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_esearch_pmids(payload: dict[str, Any] | str) -> list[str]:
    """Extract PMID list from an esearch JSON response."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    result = data.get("esearchresult") or {}
    return [str(pmid) for pmid in result.get("idlist") or []]


def parse_esearch_count(payload: dict[str, Any] | str) -> int:
    """Extract hit count from an esearch JSON response."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    result = data.get("esearchresult") or {}
    try:
        return int(result.get("count") or 0)
    except (TypeError, ValueError):
        return 0


def _parse_pub_date_from_xml(pub_date_el: ET.Element | None) -> date | None:
    if pub_date_el is None:
        return None
    year_s = pub_date_el.findtext("Year")
    if not year_s:
        medline = pub_date_el.findtext("MedlineDate")
        if medline:
            year_s = medline[:4]
    if not year_s or not year_s.isdigit():
        return None
    year = int(year_s)
    month_s = pub_date_el.findtext("Month") or "1"
    day_s = pub_date_el.findtext("Day") or "1"
    month = _MONTHS.get(month_s[:3]) if not month_s.isdigit() else int(month_s)
    if month is None:
        month = 1
    try:
        day = int(day_s)
    except ValueError:
        day = 1
    try:
        return date(year, month, day)
    except ValueError:
        return date(year, month, 1)


def _abstract_text(article: ET.Element) -> str | None:
    parts: list[str] = []
    for node in article.findall(".//Abstract/AbstractText"):
        label = node.get("Label")
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    if not parts:
        return None
    return "\n".join(parts)


def _authors(article: ET.Element) -> list[str]:
    out: list[str] = []
    for author in article.findall(".//AuthorList/Author"):
        collective = author.findtext("CollectiveName")
        if collective:
            out.append(collective.strip())
            continue
        last = (author.findtext("LastName") or "").strip()
        initials = (author.findtext("Initials") or "").strip()
        name = f"{last} {initials}".strip()
        if name:
            out.append(name)
    return out


def _mesh_terms(article: ET.Element) -> list[str]:
    terms: list[str] = []
    for heading in article.findall(".//MeshHeadingList/MeshHeading"):
        descriptor = heading.findtext("DescriptorName")
        if descriptor:
            terms.append(descriptor.strip())
    return terms


def _doi(article: ET.Element) -> str | None:
    for article_id in article.findall(".//ArticleIdList/ArticleId"):
        if article_id.get("IdType") == "doi" and article_id.text:
            return article_id.text.strip()
    return None


def parse_efetch_xml(xml_text: str) -> list[PubMedDocument]:
    """Parse efetch XML into PubMedDocument rows."""
    root = ET.fromstring(xml_text)
    docs: list[PubMedDocument] = []
    for article in root.findall("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        if not pmid:
            continue
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else None
        journal = article.findtext(".//Journal/Title")
        pub_date = _parse_pub_date_from_xml(article.find(".//Journal/JournalIssue/PubDate"))
        docs.append(
            PubMedDocument(
                pmid=str(pmid),
                title=title or None,
                abstract=_abstract_text(article),
                authors=_authors(article),
                journal=journal,
                pub_date=pub_date,
                mesh_terms=_mesh_terms(article),
                doi=_doi(article),
                raw_json={"pmid": str(pmid)},
            )
        )
    return docs


def parse_esummary_json(payload: dict[str, Any] | str) -> list[PubMedDocument]:
    """Parse esummary JSON into lightweight PubMedDocument rows (no abstract)."""
    data = json.loads(payload) if isinstance(payload, str) else payload
    result = data.get("result") or {}
    uids = [str(u) for u in result.get("uids") or []]
    docs: list[PubMedDocument] = []
    for uid in uids:
        row = result.get(uid)
        if not isinstance(row, dict):
            continue
        authors = [
            a.get("name", "").strip()
            for a in (row.get("authors") or [])
            if isinstance(a, dict) and a.get("name")
        ]
        pub_date: date | None = None
        sortpub = row.get("sortpubdate") or row.get("pubdate")
        if isinstance(sortpub, str) and len(sortpub) >= 4 and sortpub[:4].isdigit():
            # sortpubdate like "2026/07/15 00:00"
            parts = sortpub.replace("-", "/").split()[0].split("/")
            try:
                y = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 1
                d = int(parts[2]) if len(parts) > 2 else 1
                pub_date = date(y, m, d)
            except ValueError:
                pub_date = None
        elocation = str(row.get("elocationid") or "")
        doi = elocation if elocation.startswith("10.") else None
        docs.append(
            PubMedDocument(
                pmid=uid,
                title=row.get("title"),
                abstract=None,
                authors=authors,
                journal=row.get("fulljournalname") or row.get("source"),
                pub_date=pub_date,
                mesh_terms=[],
                doi=doi,
                raw_json=row,
            )
        )
    return docs
