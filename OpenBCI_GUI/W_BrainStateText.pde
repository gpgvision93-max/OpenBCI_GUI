////////////////////////////////////////////////////
//
//    W_BrainStateText.pde (ie "Brain State Text Widget")
//
//    Pipeline: Raw Data → Preprocessing → Transformer → Text Output
//
//    This widget reads preprocessed EEG data (filtered via the
//    DataProcessing pipeline), applies a transformer (band power
//    analysis with exponential smoothing) to compute per-band
//    power features, and renders a human-readable text description
//    of the user's current brain state.
//
//    Frequency bands and corresponding brain states:
//      Delta  (1–4 Hz)   — Deep Sleep / Unconscious
//      Theta  (4–8 Hz)   — Drowsy / Meditative / Creative
//      Alpha  (8–13 Hz)  — Relaxed / Calm / Reflective
//      Beta   (13–30 Hz) — Alert / Focused / Active
//      Gamma  (30–55 Hz) — Peak Performance / Hyperactive
//
//    Created: 2024
//
////////////////////////////////////////////////////

class W_BrainStateText extends Widget {

    // --- Band metadata -------------------------------------------------------
    private final String[] BAND_NAMES   = {"Delta",   "Theta",         "Alpha",              "Beta",             "Gamma"};
    private final String[] BAND_RANGES  = {"1–4 Hz",  "4–8 Hz",        "8–13 Hz",            "13–30 Hz",         "30–55 Hz"};
    private final String[] BAND_STATES  = {
        "Deep Sleep / Unconscious",
        "Drowsy / Meditative / Creative",
        "Relaxed / Calm / Reflective",
        "Alert / Focused / Active",
        "Peak Performance / Hyperactive"
    };

    // Distinct colours for each band
    private final color DELTA_COLOR = color(68,  68,  204);   // blue
    private final color THETA_COLOR = color(68,  170,  68);   // green
    private final color ALPHA_COLOR = color(200, 180,   0);   // amber
    private final color BETA_COLOR  = color(204,  68,  68);   // red
    private final color GAMMA_COLOR = color(170,  68, 170);   // purple
    private final color[] BAND_COLORS = {DELTA_COLOR, THETA_COLOR, ALPHA_COLOR, BETA_COLOR, GAMMA_COLOR};

    // --- Transformer state ---------------------------------------------------
    // Exponential smoothing factor: 0.80 balances noise suppression (higher = smoother)
    // against responsiveness to genuine state changes (lower = faster to track).
    // At a typical 250 Hz sample rate with ~16 Hz update calls this gives a time
    // constant of roughly 5 update cycles (~300 ms), long enough to avoid flickering
    // yet short enough to reflect meaningful transitions within a couple of seconds.
    private final float SMOOTH_FACTOR = 0.80f;
    private float[]   smoothedPower  = new float[5];
    private float[]   bandPower      = new float[5];
    private float     totalPower     = 0f;
    private int       dominantBand   = 2;   // default: Alpha
    private boolean   hasData        = false;

    // --- Layout constants ----------------------------------------------------
    private final int PAD         = 10;
    private final int BAR_HEIGHT  = 14;
    private final int LINE_H      = BAR_HEIGHT + 6;
    private final int LABEL_W     = 56;   // reserved width for band name labels
    // Minimum power floor prevents division-by-zero when no signal is present
    private final float MIN_POWER_THRESHOLD = 0.0001f;

    private DecimalFormat df = new DecimalFormat("0.000");

    // -------------------------------------------------------------------------
    W_BrainStateText(PApplet _parent) {
        super(_parent);
    }

    // -------------------------------------------------------------------------
    public void update() {
        super.update();
    }

    // -------------------------------------------------------------------------
    public void draw() {
        super.draw();
        drawBrainStateText();
    }

    // -------------------------------------------------------------------------
    public void screenResized() {
        super.screenResized();
    }

    // -------------------------------------------------------------------------
    public void mousePressed() {
        super.mousePressed();
    }

    // -------------------------------------------------------------------------
    public void mouseReleased() {
        super.mouseReleased();
    }

    // =========================================================================
    //  Transformer: called from DataProcessing every data cycle
    //  Reads preprocessed band-power averages, applies exponential smoothing,
    //  and determines the dominant brain state.
    // =========================================================================
    public void updateBrainStateTextData() {
        // Guard: headWidePower is only valid once at least one data cycle ran
        if (dataProcessing == null || dataProcessing.headWidePower == null) return;

        totalPower = 0f;
        for (int i = 0; i < 5; i++) {
            float raw = max(0f, dataProcessing.headWidePower[i]);
            // Exponential smoothing (low-pass in time)
            smoothedPower[i] = SMOOTH_FACTOR * smoothedPower[i] + (1f - SMOOTH_FACTOR) * raw;
            bandPower[i]     = smoothedPower[i];
            totalPower      += bandPower[i];
        }

        // Find the dominant band
        dominantBand = 0;
        for (int i = 1; i < 5; i++) {
            if (bandPower[i] > bandPower[dominantBand]) dominantBand = i;
        }

        hasData = currentBoard.isStreaming() && totalPower > 0f;
    }

    // =========================================================================
    //  Text Output renderer
    // =========================================================================
    private void drawBrainStateText() {
        pushStyle();
        noStroke();

        int cx = x + PAD;
        int cy = y + PAD;
        int cw = w - PAD * 2;

        // -- Waiting for data -------------------------------------------------
        if (!hasData) {
            fill(OPENBCI_DARKBLUE);
            textFont(h3);
            textAlign(CENTER, CENTER);
            text("Waiting for data...", x + w / 2, y + h / 2);
            popStyle();
            return;
        }

        // -- Dominant brain state header --------------------------------------
        textAlign(LEFT, TOP);
        fill(BAND_COLORS[dominantBand]);
        textFont(h2);
        text("Brain State:  " + BAND_NAMES[dominantBand], cx, cy);
        cy += 26;

        fill(OPENBCI_DARKBLUE);
        textFont(h3);
        text(BAND_STATES[dominantBand], cx, cy);
        cy += 22;

        fill(color(100, 100, 100));
        textFont(p5);
        text("Dominant band: " + BAND_RANGES[dominantBand], cx, cy);
        cy += 20;

        // -- Separator --------------------------------------------------------
        stroke(color(200, 200, 200));
        strokeWeight(1);
        line(cx, cy, cx + cw, cy);
        noStroke();
        cy += 8;

        // -- Per-band power bars ----------------------------------------------
        fill(OPENBCI_DARKBLUE);
        textFont(p4);
        textAlign(LEFT, TOP);
        text("Band Power (µV²/Hz):", cx, cy);
        cy += 18;

        int barAreaW = cw - LABEL_W - 10;
        float maxPow = MIN_POWER_THRESHOLD; // avoid div-by-zero
        for (float p : bandPower) if (p > maxPow) maxPow = p;

        for (int i = 0; i < 5; i++) {
            // Band name label
            textFont(p5);
            textAlign(LEFT, TOP);
            fill(BAND_COLORS[i]);
            text(BAND_NAMES[i], cx, cy + 2);

            // Background track
            fill(color(220, 220, 220));
            rect(cx + LABEL_W, cy, barAreaW, BAR_HEIGHT, 3);

            // Filled portion
            float fraction  = bandPower[i] / maxPow;
            int   filledW   = (int)(fraction * barAreaW);
            fill(BAND_COLORS[i]);
            if (filledW > 0) rect(cx + LABEL_W, cy, filledW, BAR_HEIGHT, 3);

            // Numeric value (right of bar)
            fill(OPENBCI_DARKBLUE);
            textFont(p5);
            textAlign(LEFT, TOP);
            text(df.format(bandPower[i]), cx + LABEL_W + barAreaW + 5, cy + 2);

            cy += LINE_H;
        }

        cy += 4;

        // -- Separator --------------------------------------------------------
        stroke(color(200, 200, 200));
        strokeWeight(1);
        line(cx, cy, cx + cw, cy);
        noStroke();
        cy += 8;

        // -- Relative power stacked bar ---------------------------------------
        fill(OPENBCI_DARKBLUE);
        textFont(p4);
        textAlign(LEFT, TOP);
        text("Relative Power:", cx, cy);
        cy += 18;

        if (totalPower > 0f) {
            int segX = cx;
            for (int i = 0; i < 5; i++) {
                int segW = (int)((bandPower[i] / totalPower) * cw);
                fill(BAND_COLORS[i]);
                rect(segX, cy, segW, BAR_HEIGHT);
                segX += segW;
            }
        }
        cy += BAR_HEIGHT + 5;

        // Legend for stacked bar
        textFont(p5);
        int legX = cx;
        for (int i = 0; i < 5; i++) {
            float pct = (totalPower > 0f) ? (bandPower[i] / totalPower) * 100f : 0f;
            if (pct >= 4f && legX + 80 <= cx + cw) {
                fill(BAND_COLORS[i]);
                textAlign(LEFT, TOP);
                text(BAND_NAMES[i] + " " + nf(pct, 0, 1) + "%", legX, cy);
                legX += 82;
            }
        }

        popStyle();
    }

}; // end class W_BrainStateText
