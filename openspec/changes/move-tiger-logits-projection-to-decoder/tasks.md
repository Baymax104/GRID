## 1. Decoder Logits Output

- [x] 1.1 Update `TigerDecoder.forward()` to project decoder hidden states through decoder-owned `decoder_mlp`.
- [x] 1.2 Make `TigerDecoder.forward()` return target-aligned raw logits with shape `(batch_size, sequence_length, codebook_size)`.
- [x] 1.3 Update `TigerDecoder.forward()` type/docstring to describe logits output.

## 2. TIGER Loss Boundary

- [x] 2.1 Update `Tiger.forward()` docstring/type expectations to reflect logits output from the decoder.
- [x] 2.2 Update `Tiger._compute_loss()` to consume logits and target IDs only.
- [x] 2.3 Remove direct `self.decoder.decoder_mlp` access from `Tiger`.

## 3. Validation

- [x] 3.1 Scan for stale `decoder_mlp` access outside `TigerDecoder`.
- [x] 3.2 Run focused lint for `src/recommendation/tiger`.
- [x] 3.3 Run OpenSpec validation for `move-tiger-logits-projection-to-decoder`.
- [x] 3.4 Run TIGER Hydra compose/import smoke.
