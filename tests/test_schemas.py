"""Tests for Qdrant collection configuration."""

from legal_agent.db.schemas import _dense_vector_params


def test_dense_vector_params_match_documented_hnsw_tuning():
    params = _dense_vector_params(768)

    assert params.size == 768
    assert params.hnsw_config.m == 24
    assert params.hnsw_config.payload_m == 24
    assert params.hnsw_config.ef_construct == 256
