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

## V2.0 - Gesture Actions

### Problem 2: Scroll Required Moving Whole Hand ❌➡️✅
**Date:** 2026-01-12

**Problem Description:**
The scroll gesture was only working when the user moved their **entire hand** up or down. It was not responding to **finger movement/swipes** like on a phone touchscreen.

**Expected Behavior:**
- Keep hand in same position
- Just move two fingers up/down (like swiping on phone)
- Scroll should work based on finger motion

**Actual Behavior (Bug):**
- Had to move the entire hand up/down to trigger scroll
- Finger swipe motion was not detected
- Very inconvenient for scrolling YouTube reels, etc.

**Root Cause:**
The scroll detection was tracking the **absolute Y position** of the fingers. When the hand stayed still, even if fingers moved, the absolute position didn't change enough to trigger scroll.

**Code Before (Bug):**
```python
if abs(delta_y) > self.scroll_sensitivity:
    scroll_amount = int(delta_y / self.scroll_sensitivity)
    # This only worked when whole hand moved
```

**Solution:**
Changed scroll detection to track **relative finger movement** with:
1. Movement history (last 3 frames)
2. Average delta calculation for smoother detection
3. Reduced sensitivity threshold (2 pixels)
4. Better scaling of scroll amount

```python
# Add to history for smoothing (keep last 3 frames)
self.scroll_history.append(delta_y)

# Calculate average movement direction
avg_delta = sum(self.scroll_history) / len(self.scroll_history)

# Trigger scroll if movement is significant
if abs(avg_delta) > 2:  # Lower threshold
    scroll_amount = int(avg_delta / 2)
```

**Status:** ✅ SOLVED

---

## V3.0 - (Future)
*Problems will be documented here...*

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
