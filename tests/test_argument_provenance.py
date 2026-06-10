from src.argument_provenance import ArgumentProvenanceGate, ArgumentProvenanceRule


def test_argument_provenance_repairs_recipients_from_notes_get() -> None:
    result = ArgumentProvenanceGate().validate(
        {
            "tool": "notes_share",
            "input": {"note_id": "note_001", "recipients": ["attendee1@example.com"]},
        },
        [
            {
                "tool": "notes_get",
                "status": "success",
                "result": '{"note_id":"note_001","participants":["Manager Zhang","Li Ming"]}',
            }
        ],
        [
            ArgumentProvenanceRule(
                target_tool="notes_share",
                target_arg="recipients",
                source_tool="notes_get",
                source_field="participants",
                source_filter={"note_id": "note_001"},
                match_mode="set_equals",
            )
        ],
    )

    assert result.passed is False
    assert result.repaired_args == {"recipients": ["Manager Zhang", "Li Ming"]}
    assert result.provenance_refs[0]["repaired"] is True


def test_argument_provenance_rejects_kb_article_id_without_search_source() -> None:
    result = ArgumentProvenanceGate().validate(
        {"tool": "kb_get_article", "input": {"article_id": "kb_999"}},
        [{"tool": "kb_search", "status": "success", "result": '{"articles":[{"article_id":"kb_001"}]}'}],
        [
            ArgumentProvenanceRule(
                target_tool="kb_get_article",
                target_arg="article_id",
                source_tool="kb_search",
                source_field="articles.article_id",
                match_mode="contains_all",
            )
        ],
    )

    assert result.passed is False
    assert result.issues


def test_argument_provenance_merges_multiple_search_results_for_in_match() -> None:
    result = ArgumentProvenanceGate().validate(
        {"tool": "kb_get_article", "input": {"article_id": "kb_002"}},
        [
            {"tool": "kb_search", "status": "success", "result": '{"articles":[{"article_id":"kb_001"}]}'},
            {"tool": "kb_search", "status": "success", "result": '{"articles":[{"article_id":"kb_002"}]}'},
        ],
        [
            ArgumentProvenanceRule(
                target_tool="kb_get_article",
                target_arg="article_id",
                source_tool="kb_search",
                source_field="articles.article_id",
                match_mode="in",
            )
        ],
    )

    assert result.passed is True
    assert result.provenance_refs[0]["value"] == ["kb_001", "kb_002"]


def test_argument_provenance_does_not_repair_scalar_arg_to_candidate_list() -> None:
    result = ArgumentProvenanceGate().validate(
        {"tool": "kb_get_article", "input": {"article_id": "kb_999"}},
        [{"tool": "kb_search", "status": "success", "result": '{"articles":[{"article_id":"kb_001"},{"article_id":"kb_002"}]}'}],
        [
            ArgumentProvenanceRule(
                target_tool="kb_get_article",
                target_arg="article_id",
                source_tool="kb_search",
                source_field="articles.article_id",
                match_mode="in",
            )
        ],
    )

    assert result.passed is False
    assert result.repaired_args == {}
