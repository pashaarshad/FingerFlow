# FingerFlow - Problems Faced & Solutions

A log of all problems encountered during development and their solutions.

---

## V1.0 - Basic Cursor Movement

### Problem 1: Mirror Cursor Movement ❌➡️✅
**Date:** 2026-01-12

**Problem Description:**
When moving the hand/finger to the **right side**, the cursor was moving to the **left side** (and vice versa). This was counter-intuitive and not user-friendly.

**Expected Behavior:**
- Move finger RIGHT → Cursor moves RIGHT
- Move finger LEFT → Cursor moves LEFT

**Actual Behavior (Bug):**
- Move finger RIGHT → Cursor moves LEFT ❌
- Move finger LEFT → Cursor moves RIGHT ❌

**Root Cause:**
The frame was already being flipped horizontally (mirror effect for the camera preview), but the coordinate mapping was ALSO inverting the X-axis. This caused a double-inversion, resulting in opposite movement.

**Code Before (Bug):**
```python
# Map to screen coordinates (mirror x-axis for natural movement)
screen_x = np.interp(x, [x_min, x_max], [self.screen_w, 0])  # Inverted!
```

**Solution:**
Changed the X-axis mapping to direct (not inverted) since the frame flip already handles the mirroring:

```python
# Map to screen coordinates (direct mapping for natural movement)
# Since frame is already flipped, no need to invert X-axis
screen_x = np.interp(x, [x_min, x_max], [0, self.screen_w])  # Direct mapping
```

**Status:** ✅ SOLVED

---

## V2.0 - (Next Version)
*Coming soon...*

---

## Template for New Problems

### Problem X: [Title]
**Date:** YYYY-MM-DD

**Problem Description:**
[Describe the problem]

**Expected Behavior:**
[What should happen]

**Actual Behavior (Bug):**
[What was happening incorrectly]

**Root Cause:**
[Why the bug occurred]

**Solution:**
[How it was fixed]

**Status:** ✅ SOLVED / 🔄 IN PROGRESS / ❌ NOT SOLVED
