import types

import pytest
import torch
from test_tiger_prefix_trace import create_trace_tiger

from src.recommendation.tiger.decoder import TigerDecoder


def legacy_check(self, prefix, batch_size=100000):
    return (self.semantic_ids.to(prefix.device)[:, None, :prefix.shape[1]] == prefix[None]).all(2).any(0)


def make_decoder(catalog, radix):
    decoder = TigerDecoder.__new__(TigerDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.semantic_ids = catalog
    decoder.codebook_size = radix
    decoder._prefix_index_source = None
    decoder._prefix_index_signature = None
    decoder._prefix_index_keys = {}
    decoder._prefix_index_safe = False
    return decoder


@pytest.mark.parametrize('radix,depth', [(1, 4), (3, 5), (256, 4), (2, 63), (2, 64)])
def test_membership_matches_token_comparison(radix, depth):
    generator = torch.Generator().manual_seed(42)
    catalog = torch.randint(radix, (23, depth), generator=generator)
    catalog = torch.cat([catalog.flip(0), catalog[:3]])
    decoder = make_decoder(catalog, radix)
    for width in range(depth + 1):
        candidates = torch.cat([
            catalog[:7, :width],
            torch.randint(-1, radix + 1, (31, width), generator=generator),
        ])
        assert torch.equal(decoder._check_valid_prefix(candidates, batch_size=7), legacy_check(decoder, candidates))


@pytest.mark.parametrize('catalog', [torch.empty(0, 2, dtype=torch.long), torch.tensor([[0, 1], [1, 0]])])
def test_empty_queries_and_catalog(catalog):
    decoder = make_decoder(catalog, 4)
    for width in (0, 1, 2):
        query = torch.empty(0, width, dtype=torch.long)
        assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))
    query = torch.tensor([[0, 1], [1, 0]])
    assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))


def test_invalid_tokens_do_not_alias_and_float_queries_preserve_comparison():
    decoder = make_decoder(torch.tensor([[1, 0], [0, 0]]), 4)
    for query in (torch.tensor([[0, 4], [-1, 4], [1, 0]]), torch.tensor([[1., 0.], [1.5, 0.]])):
        assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))
    decoder.semantic_ids = torch.tensor([[0, 4], [-1, 4]])
    query = torch.tensor([[0, 4], [-1, 4], [1, 0], [0, 0]])
    assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))


def test_noncontiguous_queries_and_catalog():
    catalog = torch.tensor([[0, 9, 1, 9], [1, 9, 0, 9]])[:, ::2]
    decoder = make_decoder(catalog, 4)
    assert torch.equal(decoder._check_valid_prefix(catalog), legacy_check(decoder, catalog))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool])
def test_integer_dtypes(dtype):
    catalog = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    decoder = make_decoder(catalog, 256)
    query = torch.tensor([[0, 1], [1, 1]], dtype=dtype)
    assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))


def test_invalid_query_shape_and_chunk_size():
    decoder = make_decoder(torch.tensor([[0, 0]]), 4)
    for query in (torch.tensor([0, 0]), torch.tensor([[0, 0, 0]])):
        with pytest.raises(ValueError, match="prefix must be a matrix"):
            decoder._check_valid_prefix(query)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        decoder._check_valid_prefix(torch.tensor([[0, 0]]), batch_size=0)


def test_cache_reuse_and_catalog_invalidation():
    decoder = make_decoder(torch.tensor([[0, 0], [1, 1]]), 4)
    query = torch.tensor([[0, 0], [2, 2]])
    decoder._check_valid_prefix(query)
    keys = decoder._prefix_index_keys[2]
    decoder._check_valid_prefix(query)
    assert decoder._prefix_index_keys[2] is keys
    decoder.semantic_ids[0] = 2
    assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))
    decoder.semantic_ids = torch.tensor([[0, 0]])
    assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))
    decoder.codebook_size = 2
    assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))


def test_inference_tensor_updates_without_version_counter():
    with torch.inference_mode():
        decoder = make_decoder(torch.tensor([[0, 0]]), 4)
        query = torch.tensor([[0, 0], [1, 1]])
        decoder._check_valid_prefix(query)
        decoder.semantic_ids[0] = 1
        assert torch.equal(decoder._check_valid_prefix(query), legacy_check(decoder, query))


def test_checkpoint_and_full_generation_remain_identical():
    torch.manual_seed(7)
    model = create_trace_tiger()
    original_state = {key: value.clone() for key, value in model.state_dict().items()}
    inputs = torch.tensor([[0, 1], [1, 0], [0, 0]])
    mask = torch.ones_like(inputs)
    with torch.inference_mode():
        indexed = model.generate(mask, inputs)
        model.decoder._check_valid_prefix = types.MethodType(legacy_check, model.decoder)
        legacy = model.generate(mask, inputs)
    assert all(torch.equal(left, right) for left, right in zip(indexed, legacy, strict=True))
    assert model.state_dict().keys() == original_state.keys()
    model.load_state_dict(original_state, strict=True)
