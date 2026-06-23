# Firmware reward patch for sense-pong (`pong.c`)

Apply on the `development` branch of https://github.com/RakiDelmoro/sense-pong
You flash this on your Windows host with `idf.py -p COMx flash monitor`.

A ready-to-apply git patch is in `firmware_reward.patch` (same directory).
If `git apply firmware_reward.patch` fails (whitespace/context drift), apply
the three changes below by hand — they're small and exact.

## The concept (why this is needed)

Right now the firmware does this when the ball escapes a side:

```c
    /* Ball escaped a side: just reset, no score */
    if (g.ball_x + BALL_SIZE < 0) {
        reset_ball(+1);
    } else if (g.ball_x > DISP_W) {
        reset_ball(-1);
    }
```

It silently resets the ball. There is **no score** and **no signal to the
outside world**. So the RL agent has no way to know when a point happens —
which means it has no reward to learn from.

We add three things, all in `pong.c`:

1. **Keep a score** (`score_left` / `score_right`).
2. **Print a one-line message to the USB serial port every time a point
   happens** — e.g. `P L 1 0`. This is the reward signal. It goes out on the
   ESP32-S3's USB-CDC, the **same COM port** `idf.py monitor` already uses, so
   you'll literally see the lines scroll by in your monitor as points happen.
3. **Show the score on screen** (two labels) so you can watch the game.

That's it. The Windows host then reads those `P L` / `P R` lines and forwards
a +1 / -1 to the container as the reward (see DESIGN.md "Reward"). The camera
is never used for reward — only the game's own score, exactly like the
reference physical-atari-rlc design.

## Reward direction (important — get this right)

The **agent controls the LEFT paddle**. The CPU controls the RIGHT paddle.

| Ball escapes | Meaning | Who scores | Reward for agent | Line emitted |
|---|---|---|---|---|
| LEFT edge (`ball_x + BALL_SIZE < 0`) | agent (left) missed | RIGHT (CPU) | **-1** | `P R <l> <r>` |
| RIGHT edge (`ball_x > DISP_W`) | CPU (right) missed | LEFT (agent) | **+1** | `P L <l> <r>` |

`<l>` and `<r>` are the new totals (score_left, score_right). The host extracts
the +1/-1 from the L/R letter; the numbers are for display/debug.

Note the existing `reset_ball` direction is already correct and we leave it:
after the agent misses (ball escaped left) the ball serves toward the right
(`reset_ball(+1)`); after the CPU misses it serves toward the left
(`reset_ball(-1)`). Normal Pong serve-after-point.

## Change 1 — add score fields to the state struct

Find this in `pong.c` (inside `typedef struct { ... } pong_state_t;`):

```c
    int16_t right_paddle_y;        /* CPU-controlled right paddle top Y */
    int16_t left_paddle_y;         /* player-controlled left paddle top Y */

    pong_phase_t state;
```

Change to:

```c
    int16_t right_paddle_y;        /* CPU-controlled right paddle top Y */
    int16_t left_paddle_y;         /* player-controlled left paddle top Y */

    int score_left;                /* RL agent score (left paddle) */
    int score_right;               /* CPU score (right paddle) */
    lv_obj_t *score_left_label;    /* on-screen score labels */
    lv_obj_t *score_right_label;

    pong_phase_t state;
```

## Change 2 — make sure there is a TAG for ESP_LOGI

Near the top of `pong.c`, after the includes, add (skip if a TAG already
exists):

```c
#include "esp_log.h"
static const char *TAG = "PONG";
```

(Using `ESP_LOGI` instead of raw `printf` guarantees the line appears on the
USB-CDC console — you already see `ESP_LOGI` lines in `idf.py monitor`, so the
console path is set up. The host ignores the `I (time) PONG:` prefix and just
regex-searches each line for `P ([LR]) (\d+) (\d+)`.)

## Change 3 — create + zero the score labels in build_game()

In `build_game()`, find:

```c
    /* PAUSE button (top center, clear of the round-screen clipping) */
    make_button(scr, 100, 32, (DISP_W - 100) / 2, 8, "PAUSE", pause_btn_cb);

    /* Ball */
    g.ball = make_rect(scr, BALL_SIZE, BALL_SIZE, 0, 0);
```

Change to:

```c
    /* PAUSE button (top center, clear of the round-screen clipping) */
    make_button(scr, 100, 32, (DISP_W - 100) / 2, 8, "PAUSE", pause_btn_cb);

    /* Score labels (left = agent, right = CPU). Reset to 0 each new game. */
    g.score_left = 0;
    g.score_right = 0;
    g.score_left_label = make_label(scr, "0", FONT_MENU, lv_color_white());
    lv_obj_align(g.score_left_label, LV_ALIGN_TOP_LEFT, 60, 8);
    g.score_right_label = make_label(scr, "0", FONT_MENU, lv_color_white());
    lv_obj_align(g.score_right_label, LV_ALIGN_TOP_RIGHT, -60, 8);

    /* Ball */
    g.ball = make_rect(scr, BALL_SIZE, BALL_SIZE, 0, 0);
```

## Change 4 — award a point + emit the reward line on escape

Find the escape block (currently "just reset, no score"):

```c
    /* Ball escaped a side: just reset, no score */
    if (g.ball_x + BALL_SIZE < 0) {
        reset_ball(+1);
    } else if (g.ball_x > DISP_W) {
        reset_ball(-1);
    }
```

Replace with:

```c
    /* Ball escaped a side: award a point, emit reward over USB-CDC, reset.
     * Left paddle (agent) guards the LEFT edge; right paddle (CPU) guards RIGHT.
     * Ball escapes LEFT  -> agent missed -> point for right -> reward -1 -> "P R".
     * Ball escapes RIGHT -> CPU missed   -> point for left  -> reward +1 -> "P L".
     * The host parses "P <L|R> <l> <r>" from the ESP32 USB-CDC console. */
    if (g.ball_x + BALL_SIZE < 0) {
        g.score_right++;
        if (g.score_right_label)
            lv_label_set_text_fmt(g.score_right_label, "%d", g.score_right);
        ESP_LOGI(TAG, "P R %d %d", g.score_left, g.score_right);
    } else if (g.ball_x > DISP_W) {
        g.score_left++;
        if (g.score_left_label)
            lv_label_set_text_fmt(g.score_left_label, "%d", g.score_left);
        ESP_LOGI(TAG, "P L %d %d", g.score_left, g.score_right);
    }
```

Note: I removed the `reset_ball(...)` calls from inside the `if` arms — see
Change 5.

## Change 5 — always reset the ball after a point (one reset call)

The original called `reset_ball` inside each arm. To keep the serve direction
and avoid forgetting to reset, put a single reset after the block. Immediately
after the block from Change 4, add:

```c
    if (g.ball_x + BALL_SIZE < 0 || g.ball_x > DISP_W) {
        /* serve toward the player who was just scored on */
        reset_ball(g.ball_x + BALL_SIZE < 0 ? +1 : -1);
    }
```

(So: ball escaped LEFT  -> reset_ball(+1) serves right; ball escaped RIGHT
-> reset_ball(-1) serves left. Same behaviour as before, just one place.)

## Change 6 — NULL the label pointers when leaving the game

In `goto_welcome()`, find:

```c
        g.ball = NULL;
        g.left_paddle = g.right_paddle = NULL;
    }
```

Change to:

```c
        g.ball = NULL;
        g.left_paddle = g.right_paddle = NULL;
        g.score_left_label = g.score_right_label = NULL;
    }
```

(Stops dangling pointers after the game screen is deleted. The tick timer is
paused in `goto_welcome` so the labels aren't touched after this, but this
keeps it clean.)

## How to verify it works (before any RL)

1. Flash and open `idf.py -p COMx flash monitor`.
2. Press START, let the ball play.
3. Every time the ball escapes a side you should see BOTH:
   - the on-screen score label tick up, and
   - a monitor line like `I (12345) PONG: P L 1 0` (or `P R 0 1`).
4. Confirm the L/R matches who scored: ball off the RIGHT edge -> `P L`
   (agent scored, +1); ball off the LEFT edge -> `P R` (CPU scored, -1).

Once those lines appear in the monitor, the host bridge can read them and the
RL loop has its reward. Nothing else on the firmware side is needed for reward.

## What the host does with these lines (for context, not firmware)

The Windows host opens the SenseCAP's COM port (the same one `idf.py monitor`
uses) at 115200 8N1, reads lines, and for each line matching
`P ([LR]) (\d+) (\d+)`:
- `P L` -> reward_delta += 1
- `P R` -> reward_delta += -1

It accumulates `reward_delta` and bundles it into the next frame packet it
sends the container: `[u32 len][JPEG][i32 reward_delta][i32 score_l][i32 score_r]`,
then zeroes the accumulator. So obs+reward are atomic per step. This is all
host/container code — you don't touch it on the firmware side.
