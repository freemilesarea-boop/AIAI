from luber_schemas import AssetType, ErrorCode, GenerationStatus, VocalGender


def test_generation_status_values_match_spec():
    assert [s.value for s in GenerationStatus] == [
        "QUEUED",
        "STARTING",
        "GENERATING",
        "POST_PROCESSING",
        "UPLOADING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    ]


def test_terminal_statuses():
    assert GenerationStatus.COMPLETED.is_terminal
    assert GenerationStatus.FAILED.is_terminal
    assert GenerationStatus.CANCELLED.is_terminal
    assert not GenerationStatus.QUEUED.is_terminal
    assert not GenerationStatus.GENERATING.is_terminal


def test_vocal_gender_options():
    assert {v.value for v in VocalGender} == {"female", "male", "instrumental"}


def test_asset_types():
    assert {a.value for a in AssetType} == {
        "MASTER",
        "FINISHED_MASTER",
        "PREVIEW",
        "STEM",
    }


def test_error_codes_are_stable_strings():
    assert ErrorCode.OUT_OF_MEMORY.value == "OUT_OF_MEMORY"
    # Phase 15R added REFERENCE_AUDIO_UNAVAILABLE. The count is asserted
    # so a new code has to be a deliberate act: these values are stored
    # and returned to clients, which makes every addition a contract
    # change rather than an implementation detail.
    assert ErrorCode.REFERENCE_AUDIO_UNAVAILABLE.value == "REFERENCE_AUDIO_UNAVAILABLE"
    assert ErrorCode.GENERATION_HAS_DERIVED_VERSIONS.value == "GENERATION_HAS_DERIVED_VERSIONS"
    assert len(ErrorCode) == 10
