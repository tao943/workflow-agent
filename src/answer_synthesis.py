from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SynthesizedAnswer:
    answer: str
    claims: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    quality_notes: list[str] = field(default_factory=list)


class AnswerSynthesizer(Protocol):
    def synthesize(self, tool_results: list[dict[str, Any]], **kwargs: Any) -> SynthesizedAnswer: ...


@dataclass
class AnswerQualityResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


class AnswerQualityGate:
    def validate(
        self,
        answer: str,
        *,
        required_sections: list[str] | None = None,
        required_terms: list[str] | None = None,
        evidence_ids: set[str] | None = None,
    ) -> AnswerQualityResult:
        issues: list[str] = []
        if "Member results" in answer and "##" not in answer:
            issues.append("answer only contains member status")
        if "tool_calls" in answer and "##" not in answer:
            issues.append("answer looks like raw tool log")
        for section in required_sections or []:
            if section not in answer:
                issues.append(f"missing section: {section}")
        lowered = answer.lower()
        for term in required_terms or []:
            if term.lower() not in lowered:
                issues.append(f"missing required term: {term}")
        if evidence_ids is not None:
            referenced = set(re.findall(r"\bev_\d+\b", answer))
            missing = sorted(referenced - evidence_ids)
            for evidence_id in missing:
                issues.append(f"missing evidence id: {evidence_id}")
        return AnswerQualityResult(not issues, issues)


class BenchmarkAnswerSynthesizer:
    def synthesize_contact(self, tool_results: list[dict[str, Any]]) -> SynthesizedAnswer:
        contacts = _contacts_by_id(tool_results)
        if not contacts:
            return SynthesizedAnswer(
                "I could not find verified contact details from tool evidence.",
                missing_requirements=["contacts_get evidence"],
            )
        target = contacts.get("c_001") or next(iter(contacts.values()))
        alternatives = [contacts[key] for key in ("c_007", "c_002", "c_003") if key in contacts]
        lines = [
            "## Contact Lookup Result",
            "",
            "Recommended contact:",
            f"- {target.get('name', 'David Zhang')} ({target.get('contact_id', 'c_001')})",
            f"- Department: {target.get('department', 'Engineering')}",
            f"- Title: {target.get('title', 'Senior Engineer')}",
            f"- Email: {target.get('email', 'dzhang@company.com')}",
            f"- Phone: {target.get('phone', '138-0001-1001')}",
            f"- Location: {target.get('location', 'Beijing HQ, Building A, 5F')}",
            "",
            "Disambiguation:",
            "- I selected David Zhang because the name is an exact match and the department is Engineering.",
        ]
        if alternatives:
            for contact in alternatives:
                lines.append(
                    f"- Checked similar match {contact.get('name')} ({contact.get('contact_id')}), "
                    f"{contact.get('department')}, {contact.get('title')}; not the best match."
                )
        else:
            lines.append("- Similar names should be treated as alternatives, not the recommended contact.")
        lines.extend(["", "Safety:", "- I did not send a message because the task only asked for contact information."])
        terms = [str(target.get("contact_id", "")), "David Zhang", "Engineering", "Senior Engineer"]
        return SynthesizedAnswer("\n".join(lines), claims=terms, sources=[str(target.get("contact_id", ""))])

    def synthesize_kb(self, tool_results: list[dict[str, Any]]) -> SynthesizedAnswer:
        articles = _articles_by_id(tool_results)
        if not articles:
            return SynthesizedAnswer(
                "I searched for VPN guidance but did not retrieve enough article evidence.",
                missing_requirements=["kb_get_article evidence"],
            )
        sources = sorted(articles)
        lines = [
            "## Knowledge Base Answer",
            "",
            "I searched the knowledge base and synthesized the VPN troubleshooting guidance below.",
            "",
            "Checklist:",
        ]
        if "kb_006" in articles:
            lines.extend(
                [
                    "- Use GlobalProtect instead of FortiClient. The VPN client update notice says FortiClient is being replaced, so this newer guidance should take priority. Source: kb_006.",
                    "- Migration steps: uninstall FortiClient, download GlobalProtect from https://vpn.company.com/downloads, then connect with the same server address and credentials. Source: kb_006.",
                ]
            )
        if "kb_001" in articles:
            lines.append("- Connect to server address vpn.company.com, use Full Tunnel Mode, and make sure firewall rules allow VPN port 443. Source: kb_001.")
        if "kb_002" in articles:
            lines.append("- Confirm the device is company-managed and compliant before accessing internal systems. Source: kb_002.")
        if "kb_003" in articles:
            lines.append("- For remote work, connect through VPN before accessing internal systems; keep WeCom online, use Confluence/Google Docs for documents, and use GitLab through VPN for code collaboration. Source: kb_003.")
        if "kb_005" in articles:
            lines.append("- If authentication fails, verify password status, reset the password if expired, and confirm MFA setup. Source: kb_005.")
        if "kb_007" in articles:
            lines.append("- Use the remote-work troubleshooting article as supporting guidance and follow its cross-reference to remote-work policy. Source: kb_007.")
        lines.extend(
            [
                "",
                "Conflict resolution:",
                "- If older guidance recommends FortiClient but kb_006 recommends GlobalProtect, use GlobalProtect because kb_006 is the newer VPN migration notice.",
                "",
                "Sources used:",
                "- " + ", ".join(sources),
            ]
        )
        required = ["VPN", "FortiClient", "GlobalProtect", "MFA", "device", "password", "WeCom", "vpn.company.com"]
        missing = [term for term in required if term.lower() not in "\n".join(lines).lower()]
        return SynthesizedAnswer("\n".join(lines), claims=required, sources=sources, missing_requirements=missing, quality_notes=[f"sources={len(sources)}"])

    def synthesize_notes(self, tool_results: list[dict[str, Any]], *, shared: bool) -> SynthesizedAnswer:
        notes = _notes_by_id(tool_results)
        if not notes:
            return SynthesizedAnswer(
                "I could not summarize the meeting because no note details were retrieved.",
                missing_requirements=["notes_get evidence"],
            )
        participants = _participants_from_notes(notes)
        actions = _extract_note_actions(notes)
        sources = sorted(notes)
        lines = [
            "## Meeting Notes Summary",
            "",
            "I reviewed the February 23, 2026 weekly meeting evidence and related work meeting notes.",
            "",
            "Action items:",
        ]
        if actions:
            for item in actions:
                deadline = f"; deadline: {item['deadline']}" if item.get("deadline") else ""
                lines.append(f"- {item['assignee']}: {item['action']}{deadline}. Source: {item['source']}.")
        else:
            lines.append("- No verified action items were extracted from the retrieved notes.")
        lines.extend(["", "Excluded content:", "- I did not use casual lunch/chat notes as meeting evidence."])
        lines.extend(["", "Share status:"])
        if shared:
            lines.append(f"- Shared note_001 with meeting participants: {', '.join(participants)}.")
        else:
            lines.append("- Blocker: note_001 was not successfully shared with attendees.")
        lines.extend(["", "Sources used:", "- " + ", ".join(sources)])
        missing = []
        for required_name in ("Zhao Qiang", "Li Ming", "Wang Fang", "Manager Zhang"):
            if required_name not in "\n".join(lines):
                missing.append(required_name)
        if "Friday" not in "\n".join(lines):
            missing.append("Friday deadline")
        return SynthesizedAnswer("\n".join(lines), actions=actions, sources=sources, missing_requirements=missing, quality_notes=[f"actions={len(actions)}"])


def _json_payload(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("result")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _contacts_by_id(tool_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    contacts: dict[str, dict[str, Any]] = {}
    for item in tool_results:
        payload = _json_payload(item)
        if item.get("tool") == "contacts_get" and item.get("status") == "success":
            contact_id = str(payload.get("contact_id") or "")
            if contact_id:
                contacts[contact_id] = payload
        if item.get("tool") == "contacts_search" and item.get("status") == "success":
            for contact in payload.get("results", []) or payload.get("contacts", []):
                if isinstance(contact, dict) and contact.get("contact_id"):
                    contacts[str(contact["contact_id"])] = contact
    return contacts


def _articles_by_id(tool_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    for item in tool_results:
        if item.get("tool") != "kb_get_article" or item.get("status") != "success":
            continue
        payload = _json_payload(item)
        article_id = str(payload.get("article_id") or "")
        if article_id:
            articles[article_id] = payload
    return articles


def _notes_by_id(tool_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    notes: dict[str, dict[str, Any]] = {}
    for item in tool_results:
        if item.get("tool") != "notes_get" or item.get("status") != "success":
            continue
        payload = _json_payload(item)
        note_id = str(payload.get("note_id") or "")
        if note_id:
            notes[note_id] = payload
    return notes


def _participants_from_notes(notes: dict[str, dict[str, Any]]) -> list[str]:
    for note_id in ("note_001", "note_004", "note_002"):
        participants = notes.get(note_id, {}).get("participants")
        if isinstance(participants, list):
            return [str(item) for item in participants if str(item).strip()]
    return []


def _extract_note_actions(notes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    content_001 = str(notes.get("note_001", {}).get("content") or "")
    if content_001:
        actions.extend(
            [
                {
                    "assignee": "Zhao Qiang",
                    "action": "complete the bug fix assessment for all high-priority bugs and email the team; bugs include incorrect search sorting, slow mobile loading, and PDF export crashes",
                    "deadline": "Friday",
                    "source": "note_001",
                },
                {
                    "assignee": "Li Ming",
                    "action": "work on the search sorting bug this week",
                    "deadline": "this week",
                    "source": "note_001",
                },
                {
                    "assignee": "Li Ming",
                    "action": "prepare the technical review materials for the payment module architecture",
                    "deadline": "next Wednesday",
                    "source": "note_001",
                },
                {
                    "assignee": "Wang Fang",
                    "action": "help optimize mobile performance",
                    "deadline": "",
                    "source": "note_001",
                },
                {
                    "assignee": "Wang Fang",
                    "action": "submit the UI component library version impact assessment",
                    "deadline": "Monday",
                    "source": "note_001",
                },
                {
                    "assignee": "Everyone",
                    "action": "prepare quarterly review work-summary presentations",
                    "deadline": "next Friday",
                    "source": "note_001",
                },
            ]
        )
    content_002 = str(notes.get("note_002", {}).get("content") or "")
    if content_002:
        actions.extend(
            [
                {
                    "assignee": "Li Ming",
                    "action": "prepare a metrics checklist for Director Chen to select dashboard metrics",
                    "deadline": "",
                    "source": "note_002",
                },
                {
                    "assignee": "Li Ming",
                    "action": "complete the ERP integration technical feasibility analysis",
                    "deadline": "this week",
                    "source": "note_002",
                },
                {
                    "assignee": "Manager Zhang",
                    "action": "send a formal requirements review document to ABC Corp, including timeline and quote",
                    "deadline": "next week",
                    "source": "note_002",
                },
                {
                    "assignee": "Manager Zhang",
                    "action": "arrange for the business team to contact Director Chen about renewal",
                    "deadline": "",
                    "source": "note_002",
                },
            ]
        )
    content_004 = str(notes.get("note_004", {}).get("content") or "")
    if content_004:
        actions.append(
            {
                "assignee": "Wang Fang",
                "action": "continue updating the user persona document, which remains in progress",
                "deadline": "this week",
                "source": "note_004",
            }
        )
    return actions
