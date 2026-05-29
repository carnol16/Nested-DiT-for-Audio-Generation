# NSynth Dataset Audit

**Date:** 2026-05-28  
**Dataset:** NSynth (Google Magenta)

## File Counts

| Split | Files |
|-------|-------|
| train | 289,205 |
| valid | 12,678 |
| test  | 4,096 |
| **Total** | **305,979** |

## Audio Properties

| Property | Value |
|----------|-------|
| Sample rate | 16,000 Hz (uniform across all splits — confirmed via examples.json) |
| Duration | 4.0 seconds (uniform) |
| Bit depth | 16-bit signed integer |
| Channels | Mono |
| File size | 128,044 bytes (uniform) |

## Quality Checks

- Zero-length files: **0**
- Corrupted files: **None detected** (all files are exactly 128,044 bytes)
- Sample rate consistency: **100%** — all files are 16 kHz

## Preprocessing Note

DAC expects 44,100 Hz input. Every file must be resampled from 16,000 Hz → 44,100 Hz before encoding. This is a ~2.756x upsample. The resampling must happen before DAC encode, not after decode.